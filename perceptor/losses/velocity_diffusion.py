from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import LossInterface
from perceptor import models, transforms
from perceptor.models.velocity_diffusion import diffusion_space


class VelocityDiffusion(LossInterface):
    def __init__(self, model, noise):
        super().__init__()
        self.model = model
        self.noise = nn.Parameter(noise, requires_grad=True)

    def diffuse_denoise(self, denoised, ts, **extra_kwargs):
        predictions = self.model.predictions(
            self.model.diffuse(denoised, ts, noise=self.noise),
            ts,
            **extra_kwargs,
        )
        return predictions.denoised_images

    def forward(self, denoised, ts=0.7, **extra_kwargs):
        if denoised.shape[-3:] != self.model.shape:
            denoised = transforms.resize(denoised, out_shape=self.model.shape[-2:])
        with torch.no_grad():
            predicted_denoised = self.diffuse_denoise(denoised, ts, **extra_kwargs)
        return F.mse_loss(denoised, predicted_denoised)

    @contextmanager
    def guided_resample_(
        self,
        denoised,
        from_ts,
        to_ts,
        guidance_scale=0.5,
        clamp_value=1e-6,
        **extra_kwargs
    ):
        """
        Resamples noise in direction of the gradient

        Usage:

            with diffusion.guided_resample_(images) as diffused_denoised:
                clip(diffused_denoised).backward()
        """
        with torch.enable_grad():
            from_diffused = self.model.diffuse(denoised, from_ts, noise=self.noise)
            predictions = self.model.predictions(from_diffused, from_ts, **extra_kwargs)
            diffuse_denoise = predictions.denoised_images
            yield diffuse_denoise
        guided_predictions = predictions.guided(
            -self.noise.grad, guidance_scale=guidance_scale, clamp_value=clamp_value
        )
        # diffused_images = guided_predictions.resample(
        #     to_ts,
        #     **extra_kwargs,
        # )
        self.noise.data = guided_predictions.resample_noise(
            to_ts,
            **extra_kwargs,
        )
        self.noise.grad.zero_()
        # # hack to make noise std stay around 1
        # if self.noise.std() <= 0.98:
        #     self.noise.data += (
        #         torch.randn_like(self.noise) * (1 - self.noise.std() ** 2).sqrt()
        #     )

    def compensate_noise_(self, from_denoised, to_denoised):
        old_pred = diffusion_space.encode(from_denoised)
        new_pred = diffusion_space.encode(to_denoised)
        delta_pred = new_pred - old_pred
        self.noise.data = self.noise - delta_pred

        # hack to make noise std stay around 1
        # if self.noise.std() <= 0.99:
        #     self.noise.data += (
        #         torch.randn_like(self.noise) * (1 - self.noise.std() ** 2).sqrt()
        #     )

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

    with diffusion_loss.guided_resample_(
        images, from_ts=0.7, to_ts=0.6
    ) as diffused_denoised:
        diffused_denoised.square().mean().backward()
