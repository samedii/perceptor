import torch
from torch import nn

from perceptor.transforms.interface import TransformInterface
from . import utils


class Diffusion(TransformInterface):
    """
    Diffusion sampling transform.
    """

    def __init__(self, shape, n_steps=50, from_t=1, to_t=0):
        super().__init__()
        self.n_steps = n_steps
        modified_steps = n_steps - 4 * 3 + 3  # 4 steps inside a prk step
        t = torch.linspace(from_t, to_t, modified_steps + 1)[:-1]
        self.steps = nn.Parameter(
            torch.cat([utils.get_spliced_ddpm_cosine_schedule(t), torch.zeros([1])])[
                :, None
            ]
            * torch.ones([1, shape[0]]),
            requires_grad=False,
        )
        self.plms_eps_queue = nn.Parameter(None, requires_grad=False)
        self.prk = nn.ParameterDict({})

    def __iter__(self):
        index = 0
        while len(self.steps) >= 2:
            yield index, self.t
            index += 1

    @property
    def t(self):
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

    def diffused_images(self, x):
        return (x + 1) / 2

    def noise(self, x, velocity):
        return utils.velocity_to_eps(velocity, x, self.t)

    # @staticmethod
    # def x(predicted_images, eps, t):
    #     if isinstance(t, float) or t.ndim == 0:
    #         t = torch.full((predicted_images.shape[0],), t).to(predicted_images)
    #     pred = predicted_images.mul(2).sub(1)
    #     alphas, sigmas = utils.t_to_alpha_sigma(t)
    #     return pred * alphas[:, None, None, None] + eps * sigmas[:, None, None, None]

    @property
    def alphas(self):
        alphas, _ = utils.t_to_alpha_sigma(self.t)
        return alphas

    @property
    def sigmas(self):
        _, sigmas = utils.t_to_alpha_sigma(self.t)
        return sigmas

    def encode(self, images, noise=None):
        alphas, sigmas = utils.t_to_alpha_sigma(self.t.to(images))
        if noise is None:
            noise = torch.randn_like(images)
        return (images.mul(2).sub(1) * alphas + noise * sigmas).add(1).div(2)

    def decode(self, x, velocity):
        return self.predicted_images(x, velocity)

    def predicted_images(self, x, velocity):
        alphas, sigmas = utils.t_to_alpha_sigma(self.t)
        pred = x * alphas[:, None, None, None] - velocity * sigmas[:, None, None, None]
        return (pred + 1) / 2

    def guided_velocity(self, x, velocity):
        guided_velocity = (
            velocity.detach()
            + x.grad
            * (self.sigmas[:, None, None, None] / self.alphas[:, None, None, None])
            * 500
        )
        return guided_velocity

    def forced_velocity(self, x, forced_predicted):
        replaced_pred = forced_predicted * 2 - 1
        forced_velocity = (
            x * self.alphas[:, None, None, None] - replaced_pred
        ) / self.sigmas[:, None, None, None]
        return forced_velocity

    @torch.no_grad()
    def step_(self, x, velocity):
        if len(self.plms_eps_queue) < 3:
            if "eps_1" not in self.prk:
                return self.prk_step1_(x, velocity)
            elif "eps_2" not in self.prk:
                return self.prk_step2_(x, velocity)
            elif "eps_3" not in self.prk:
                return self.prk_step3_(x, velocity)
            else:
                assert self.t == self.steps[1]
                return self.prk_step4_(x, velocity)
        else:
            return self.plms_step_(x, velocity)

    @torch.no_grad()
    def plms_step_(self, x, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        eps = utils.velocity_to_eps(velocity, x, t_1)
        eps_prime = (
            55 * eps
            - 59 * self.plms_eps_queue[-1]
            + 37 * self.plms_eps_queue[-2]
            - 9 * self.plms_eps_queue[-3]
        ) / 24
        x_new, _ = utils.transfer(self.x, eps_prime, t_1, t_2)
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue[1:], eps[None]]), requires_grad=False
        )
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return x_new

    @torch.no_grad()
    def prk_step1_(self, velocity):
        self.prk["x"] = nn.Parameter(self.x.detach().clone(), requires_grad=False)
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_1"] = nn.Parameter(
            utils.velocity_to_eps(velocity, self.x, t_1), requires_grad=False
        )
        x_1, _ = utils.transfer(self.x, self.prk["eps_1"], t_1, t_mid)
        return x_1

    @torch.no_grad()
    def prk_step2_(self, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_1 = self.x
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_2"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_1, t_mid), requires_grad=False
        )
        x_2, _ = utils.transfer(self.prk["x"], self.prk["eps_2"], t_1, t_mid)
        return x_2

    @torch.no_grad()
    def prk_step3_(self, x, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_2 = x
        t_mid = (t_2 + t_1) / 2
        self.prk["eps_3"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_2, t_mid), requires_grad=False
        )
        x_3, _ = utils.transfer(self.prk["x"], self.prk["eps_3"], t_1, t_2)
        return x_3

    @torch.no_grad()
    def prk_step4_(self, x, velocity):
        t_1 = self.steps[0]
        t_2 = self.steps[1]
        x_3 = x
        eps_4 = utils.velocity_to_eps(velocity, x_3, t_2)
        eps_prime = (
            self.prk["eps_1"] + 2 * self.prk["eps_2"] + 2 * self.prk["eps_3"] + eps_4
        ) / 6
        x_new, _ = utils.transfer(self.prk["x"], eps_prime, t_1, t_2)
        self.prk = nn.ParameterDict({})
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue, eps_prime[None]]), requires_grad=False
        )
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return x_new
