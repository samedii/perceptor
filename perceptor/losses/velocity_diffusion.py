from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import LossInterface
from perceptor import models, transforms
from perceptor.models.velocity_diffusion import diffusion_space


class VelocityDiffusion(LossInterface):
    def __init__(self, model, noise, from_ts=0.5, resample_ts=0.3):
        super().__init__()
        self.from_ts = from_ts
        self.resample_ts = resample_ts
        self.model = model
        self.noise = nn.Parameter(noise, requires_grad=True)

    def diffuse_denoise(self, denoised, **extra_kwargs):
        predictions = self.model.predictions(
            self.model.diffuse(denoised, self.from_ts, noise=self.noise),
            self.from_ts,
            **extra_kwargs,
        )
        return predictions.denoised_images

    def forward(self, images, frozen_diffused_denoised):
        return F.mse_loss(
            frozen_diffused_denoised.detach().clamp(0, 1),
            transforms.clamp_with_grad(images),
        )

    @contextmanager
    def guided_resample_(
        self, denoised, guidance_scale=0.5, clamp_value=1e-6, **extra_kwargs
    ):
        """
        Resamples noise in direction of the gradient

        Usage:

            with diffusion.guided_resample_(images) as diffused_denoised:
                clip(diffused_denoised).backward()
        """
        if self.noise.grad is not None:
            self.noise.grad.zero_()
        with torch.enable_grad():
            from_diffused = self.model.diffuse(denoised, self.from_ts, noise=self.noise)
            predictions = self.model.predictions(
                from_diffused, self.from_ts, **extra_kwargs
            )
            diffuse_denoise = predictions.denoised_images
            yield diffuse_denoise
        guided_predictions = predictions.guided(
            -self.noise.grad, guidance_scale=guidance_scale, clamp_value=clamp_value
        )
        self.noise.data = guided_predictions.resample_noise(
            self.resample_ts,
            **extra_kwargs,
        )
        self.noise.grad.zero_()

    def compensate_noise_(self, from_denoised, to_denoised):
        old_pred = diffusion_space.encode(from_denoised)
        new_pred = diffusion_space.encode(to_denoised)
        delta_pred = new_pred - old_pred
        self.noise.data = self.noise.data - delta_pred.data

    def noise_step_(self, from_denoised, from_t, to_t, to_denoised):
        """
        1. Step to to_t
        2. Add noise to from_t
        """
        from_diffused = self.model.diffuse(from_denoised, from_t, noise=self.noise)
        forced_from_eps = self.model.forced_eps(from_diffused, from_t, to_denoised)
        # should be same as self.noise?
        assert forced_from_eps == self.noise
        to_diffused = self.model.step(from_diffused, from_t, to_t, to_denoised)
        reversed_diffused = self.model.reverse_step(to_diffused, to_t, from_t)
        reversed_eps = self.model.eps(reversed_diffused, from_t)
        self.noise.data = reversed_eps


def test_velocity_diffusion_loss():
    model = models.VelocityDiffusion().cuda()
    diffusion_loss = VelocityDiffusion(model, torch.randn((1, *model.shape))).cuda()
    images = torch.zeros((1, *model.shape)).cuda()

    with diffusion_loss.guided_resample_(images) as diffused_denoised:
        diffused_denoised.square().mean().backward()
