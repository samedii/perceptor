import torch
from .clamp_with_grad import clamp_with_grad

from .interface import TransformInterface


def dynamic_threshold(images, quantile=0.95):
    denoised_xs = images.mul(2).sub(1)
    dynamic_threshold = torch.quantile(
        denoised_xs.flatten(start_dim=1).abs(), quantile, dim=1
    ).clamp(min=1.0)
    denoised_xs = (
        clamp_with_grad(
            denoised_xs,
            -dynamic_threshold,
            dynamic_threshold,
        )
        / dynamic_threshold
    )
    denoised_images = denoised_xs.add(1).div(2)
    return denoised_images


class DynamicThreshold(TransformInterface):
    def __init__(self, quantile=0.95):
        super().__init__()
        self.quantile = quantile

    def encode(self, images, quantile=None):
        return dynamic_threshold(images, quantile or self.quantile)

    def decode(self, images):
        return images
