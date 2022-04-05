import torch
from torch import nn

from perceptor.drawers.interface import DrawingInterface
from . import utils


class Diffusion(DrawingInterface):
    """
    Diffusion sampling drawer.

    Usage:

        diffusion = drawers.Diffusion(
            shape=(1, 3, 512, 512),
            n_steps=50,
        ).to(device)

        yfcc_2_model = models.VelocityDiffusion("yfcc_2").to(device)

        for iteration, x, t in tqdm(diffusion):
            velocity = yfcc_2_model(x, t)
            diffusion.step_(velocity)
            display(
                utils.pil_image(diffusion.predicted_images(velocity))
            )
    """

    def __init__(self, shape, n_steps=50, from_t=1, to_t=0, method="ddim", eta=None):
        super().__init__()
        self.method = method
        self.eta = eta
        self.n_steps = n_steps
        self.x = nn.Parameter(torch.randn(shape), requires_grad=True)
        if self.method == "ddim":
            self.n_steps = n_steps
        else:
            self.n_steps = n_steps - 4 * 3 + 3  # 4 steps inside a prk step
            self.plms_eps_queue = nn.Parameter(None, requires_grad=False)
            self.prk = nn.ParameterDict({})

        t = torch.linspace(1, 0, self.n_steps + 1)[:-1]
        steps = torch.cat([utils.get_spliced_ddpm_cosine_schedule(t), torch.zeros([1])])
        steps = steps[(steps <= from_t) & (steps >= to_t)]
        self.steps = nn.Parameter(
            steps[:, None].repeat(1, shape[0]), requires_grad=False
        )

    @staticmethod
    def from_image(image, n_steps=50, from_t=0.7, to_t=0, noise=None):
        diffusion = Diffusion(image.shape, n_steps, from_t, to_t)
        diffusion.replace_(diffusion.encode(image, noise))
        return diffusion

    def __iter__(self):
        index = 0
        while len(self.steps) >= 2:
            yield index, self.x, self.t
            index += 1

    @property
    def t(self):
        if self.method == "ddim":
            return self.steps[0]
        else:
            if len(self.plms_eps_queue) < 3:
                if "eps_1" not in self.prk:
                    return self.steps[0]
                elif "eps_2" not in self.prk:
                    return (self.steps[0] + self.steps[1]) / 2
                elif "eps_3" not in self.prk:
                    return (self.steps[0] + self.steps[1]) / 2
                else:
                    return self.steps[1]
            else:
                return self.steps[0]

    @property
    def diffused_images(self):
        return (self.x + 1) / 2

    @property
    def alphas(self):
        alphas, _ = utils.t_to_alpha_sigma(self.t)
        return alphas

    @property
    def sigmas(self):
        _, sigmas = utils.t_to_alpha_sigma(self.t)
        return sigmas

    def synthesize(self, velocity=None):
        if velocity is None:
            return self.diffused_images
        else:
            return self.predicted_images(velocity)

    def encode(self, images, noise=None):
        alphas, sigmas = utils.t_to_alpha_sigma(self.t.to(images))
        if noise is None:
            noise = torch.randn_like(images)
        return (images.mul(2).sub(1) * alphas + noise * sigmas).add(1).div(2)

    def replace_(self, diffused_images):
        self.x.data.copy_(diffused_images.mul(2).sub(1))
        return self

    def predicted_images(self, velocity):
        alphas, sigmas = utils.t_to_alpha_sigma(self.t)
        pred = (
            self.x * alphas[:, None, None, None]
            - velocity * sigmas[:, None, None, None]
        )
        return (pred + 1) / 2

    def guided_velocity(self, velocity):
        guided_velocity = (
            velocity.detach()
            + self.x.grad
            * (self.sigmas[:, None, None, None] / self.alphas[:, None, None, None])
            * 500
        )
        return guided_velocity

    def forced_velocity(self, forced_predicted):
        replaced_pred = forced_predicted * 2 - 1
        forced_velocity = (
            self.x * self.alphas[:, None, None, None] - replaced_pred
        ) / self.sigmas[:, None, None, None]
        return forced_velocity

    @torch.no_grad()
    def step_(self, velocity):
        if self.method == "ddim":
            return self.ddim_step_(velocity)
        else:
            if len(self.plms_eps_queue) < 3:
                if "eps_1" not in self.prk:
                    return self.prk_step1_(velocity)
                elif "eps_2" not in self.prk:
                    return self.prk_step2_(velocity)
                elif "eps_3" not in self.prk:
                    return self.prk_step3_(velocity)
                else:
                    assert self.t == self.steps[1]
                    return self.prk_step4_(velocity)
            else:
                return self.plms_step_(velocity)

    @torch.no_grad()
    def ddim_step_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]

        alphas, sigmas = utils.t_to_alpha_sigma(t_1)
        next_alphas, next_sigmas = utils.t_to_alpha_sigma(t_2)

        pred = (
            self.x * alphas[:, None, None, None]
            - velocity * sigmas[:, None, None, None]
        )
        eps = (
            self.x * sigmas[:, None, None, None]
            + velocity * alphas[:, None, None, None]
        )

        if self.eta is not None:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = (
                self.eta
                * (next_sigmas**2 / sigmas**2).sqrt()
                * (1 - alphas**2 / next_alphas**2).sqrt()
            )
            adjusted_sigma = (next_sigmas**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x_new = pred * next_alphas + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            x_new += torch.randn_like(x_new) * ddim_sigma
        else:
            x_new = (
                pred * next_alphas[:, None, None, None]
                + eps * next_sigmas[:, None, None, None]
            )

        self.x.copy_(x_new)
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return (pred + 1) / 2

    @torch.no_grad()
    def plms_step_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        eps = utils.velocity_to_eps(velocity, self.x, t_1)
        eps_prime = (
            55 * eps
            - 59 * self.plms_eps_queue[-1]
            + 37 * self.plms_eps_queue[-2]
            - 9 * self.plms_eps_queue[-3]
        ) / 24
        x_new, pred = utils.transfer(self.x, eps_prime, t_1, t_2)
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue[1:], eps[None]]), requires_grad=False
        )
        self.x.copy_(x_new)
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return (pred + 1) / 2

    @torch.no_grad()
    def prk_step1_(self, velocity):
        self.prk["x"] = nn.Parameter(self.x.detach().clone(), requires_grad=False)
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_1"] = nn.Parameter(
            utils.velocity_to_eps(velocity, self.x, t_1), requires_grad=False
        )
        x_1, pred = utils.transfer(self.x, self.prk["eps_1"], t_1, t_mid)
        self.x.copy_(x_1)
        return (pred + 1) / 2

    @torch.no_grad()
    def prk_step2_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_1 = self.x
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_2"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_1, t_mid), requires_grad=False
        )
        x_2, pred = utils.transfer(self.prk["x"], self.prk["eps_2"], t_1, t_mid)
        self.x.copy_(x_2)
        return (pred + 1) / 2

    @torch.no_grad()
    def prk_step3_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_2 = self.x
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_3"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_2, t_mid), requires_grad=False
        )
        x_3, pred = utils.transfer(self.prk["x"], self.prk["eps_3"], t_1, t_2)
        self.x.copy_(x_3)
        return (pred + 1) / 2

    @torch.no_grad()
    def prk_step4_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_3 = self.x
        eps_4 = utils.velocity_to_eps(velocity, x_3, t_2)
        eps_prime = (
            self.prk["eps_1"] + 2 * self.prk["eps_2"] + 2 * self.prk["eps_3"] + eps_4
        ) / 6
        x_new, pred = utils.transfer(self.prk["x"], eps_prime, t_1, t_2)
        self.x.copy_(x_new)
        self.prk = nn.ParameterDict({})
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue, eps_prime[None]]), requires_grad=False
        )
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return (pred + 1) / 2
