import torch
from torch import nn

from .interface import DrawingInterface
from perceptor import models


class DeepImagePrior(DrawingInterface):
    def __init__(self, size, n_feature_channels=64, output_channels=3):
        super().__init__()
        self.deep_image_prior = models.DeepImagePrior(
            shape=(n_feature_channels, *size), output_channels=output_channels
        )
        self.latents = nn.Parameter(
            self.deep_image_prior.random_latents(), requires_grad=False
        )
        self.images = nn.Parameter(torch.zeros((1, output_channels, *size)))

    def synthesize(self, _=None):
        return self.deep_image_prior(self.latents) + self.images

    def loss(self):
        return self.images.abs().mean().mul(0.0001)
