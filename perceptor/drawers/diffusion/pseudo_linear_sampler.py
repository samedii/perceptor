import torch
from torch import nn

from perceptor.drawers.interface import DrawingInterface
from . import utils


class PseudoLinearSampler(DrawingInterface):
    def __init__(self, n_steps=100, from_value=1, to_value=0, eta=None):
        """
        Faster sampling by keeping track of previous eps and
        using a pseudo-linear step. Slightly modified from original
        where first eps were collected with prk steps.

        Usage:
            sampler = drawers.PseudoLinearSampler(from_value=0.8).to(device)
            for iteration, from_t, to_to in sampler:
                eps = sampler.eps_(
                    diffusion_model.eps(diffused_images, from_t)
                )
                denoised = diffusion_model.denoise(diffused_images, from_t, eps)
                diffused_images = diffusion_model.step(diffused_images, denoised, from_t, to_t)
        """
        super().__init__()
        self.n_steps = n_steps
        self.plms_eps_queue = nn.Parameter(None, requires_grad=False)
        steps = (
            utils.get_spliced_ddpm_cosine_schedule(
                torch.linspace(1, 0, self.n_steps + 1)
            )
            * (from_value - to_value)
            + to_value
        )
        self.steps = nn.Parameter(steps[:, None], requires_grad=False)
        self.called_eps = 0

    def __iter__(self):
        steps = self.steps.clone()
        for index, (from_value, to_value) in enumerate(zip(steps, steps[1:])):
            if self.called_eps != index:
                raise ValueError("Noise was not updated with sampler.eps_")
            yield index, from_value, to_value

    def __len__(self):
        return len(self.steps) - 1

    def eps_(self, eps):
        self.called_eps += 1
        if len(self.plms_eps_queue) < 3:
            return self.ddim_eps_(eps)
        else:
            return self.plms_eps_(eps)

    def plms_eps_(self, eps):
        eps_prime = (
            55 * eps
            - 59 * self.plms_eps_queue[-1]
            + 37 * self.plms_eps_queue[-2]
            - 9 * self.plms_eps_queue[-3]
        ) / 24
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue[1:], eps.detach()[None]]),
            requires_grad=False,
        )
        return eps_prime

    def ddim_eps_(self, eps):
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue, eps.detach()[None]]), requires_grad=False
        )
        return eps
