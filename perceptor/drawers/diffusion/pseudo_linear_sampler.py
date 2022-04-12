import torch
from torch import nn

from perceptor.drawers.interface import DrawingInterface
from . import utils


class PseudoLinearSampler(DrawingInterface):
    def __init__(self, n_steps=100, from_t=1, to_t=0, eta=None):
        """
        Faster sampling by keeping track of previous predictions and
        using a pseudo-linear step.

        Usage:
            sampler = drawers.PseudoLinearSampler(from_t=0.8).to(device)
            for iteration, from_t in sampler:
                denoised = diffusion.denoise(images, from_t)
                images, denoised = sampler.step(images, denoised)
        """
        super().__init__()
        self.eta = eta
        self.n_steps = n_steps - 4 * 3 + 3  # 4 steps inside a prk step
        self.plms_eps_queue = nn.Parameter(None, requires_grad=False)
        self.prk = nn.ParameterDict({})
        t = torch.linspace(1, 0, self.n_steps + 1)[:-1]
        steps = torch.cat([utils.get_spliced_ddpm_cosine_schedule(t), torch.zeros([1])])
        steps = steps[(steps <= from_t) & (steps >= to_t)]
        self.steps = nn.Parameter(steps[:, None], requires_grad=False)

    def __iter__(self):
        index = 0
        while len(self.steps) >= 2:
            yield index, self.t
            index += 1

    def __len__(self):
        return self.n_steps

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

    @property
    def alphas(self):
        alphas, _ = utils.t_to_alpha_sigma(self.t)
        return alphas

    @property
    def sigmas(self):
        _, sigmas = utils.t_to_alpha_sigma(self.t)
        return sigmas

    @torch.no_grad()
    def step(self, diffused, denoised):
        if len(self.plms_eps_queue) < 3:
            if "eps_1" not in self.prk:
                return self.prk_step1_(diffused, denoised)
            elif "eps_2" not in self.prk:
                return self.prk_step2_(diffused, denoised)
            elif "eps_3" not in self.prk:
                return self.prk_step3_(diffused, denoised)
            else:
                assert self.t == self.steps[1]
                return self.prk_step4_(diffused, denoised)
        else:
            return self.plms_step_(diffused, denoised)

    @torch.no_grad()
    def plms_step_(self, diffused, denoised):
        x = diffused.mul(2).sub(1)
        pred = denoised.mul(2).sub(1)
        velocity = (x * self.alphas[:, None, None, None] - pred) / self.sigmas[
            :, None, None, None
        ]
        from_t = self.steps[0]
        to_t = self.steps[1]
        eps = utils.velocity_to_eps(velocity, x, from_t)
        eps_prime = (
            55 * eps
            - 59 * self.plms_eps_queue[-1]
            + 37 * self.plms_eps_queue[-2]
            - 9 * self.plms_eps_queue[-3]
        ) / 24
        to_x, pred = utils.transfer(x, eps_prime, from_t, to_t)
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue[1:], eps[None]]), requires_grad=False
        )
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return to_x.add(1).div(2), pred.add(1).div(2)

    @torch.no_grad()
    def prk_step1_(self, diffused, denoised):
        x = diffused.mul(2).sub(1)
        pred = denoised.mul(2).sub(1)
        velocity = (x * self.alphas[:, None, None, None] - pred) / self.sigmas[
            :, None, None, None
        ]
        self.prk["x"] = nn.Parameter(x.detach().clone(), requires_grad=False)
        from_t = self.steps[0]
        to_t = self.steps[1]
        mid_t = (to_t + from_t) / 2
        self.prk["eps_1"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x, from_t), requires_grad=False
        )
        to_x, pred = utils.transfer(x, self.prk["eps_1"], from_t, mid_t)
        return to_x.add(1).div(2), pred.add(1).div(2)

    @torch.no_grad()
    def prk_step2_(self, diffused, denoised):
        x = diffused.mul(2).sub(1)
        pred = denoised.mul(2).sub(1)
        velocity = (x * self.alphas[:, None, None, None] - pred) / self.sigmas[
            :, None, None, None
        ]
        from_t = self.steps[0]
        to_t = self.steps[1]
        x_1 = x
        mid_t = (to_t + from_t) / 2
        self.prk["eps_2"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_1, mid_t), requires_grad=False
        )
        to_x, pred = utils.transfer(self.prk["x"], self.prk["eps_2"], from_t, mid_t)
        return to_x.add(1).div(2), pred.add(1).div(2)

    @torch.no_grad()
    def prk_step3_(self, diffused, denoised):
        x = diffused.mul(2).sub(1)
        pred = denoised.mul(2).sub(1)
        velocity = (x * self.alphas[:, None, None, None] - pred) / self.sigmas[
            :, None, None, None
        ]
        from_t = self.steps[0]
        to_t = self.steps[1]
        x_2 = x
        mid_t = (to_t + from_t) / 2
        self.prk["eps_3"] = nn.Parameter(
            utils.velocity_to_eps(velocity, x_2, mid_t), requires_grad=False
        )
        to_x, pred = utils.transfer(self.prk["x"], self.prk["eps_3"], from_t, to_t)
        return to_x.add(1).div(2), pred.add(1).div(2)

    @torch.no_grad()
    def prk_step4_(self, diffused, denoised):
        x = diffused.mul(2).sub(1)
        pred = denoised.mul(2).sub(1)
        velocity = (x * self.alphas[:, None, None, None] - pred) / self.sigmas[
            :, None, None, None
        ]
        from_t = self.steps[0]
        to_t = self.steps[1]
        x_3 = x
        eps_4 = utils.velocity_to_eps(velocity, x_3, to_t)
        eps_prime = (
            self.prk["eps_1"] + 2 * self.prk["eps_2"] + 2 * self.prk["eps_3"] + eps_4
        ) / 6
        to_x, pred = utils.transfer(self.prk["x"], eps_prime, from_t, to_t)
        self.prk = nn.ParameterDict({})
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue, eps_prime[None]]), requires_grad=False
        )
        self.steps = nn.Parameter(self.steps[1:], requires_grad=False)
        return to_x.add(1).div(2), pred.add(1).div(2)
