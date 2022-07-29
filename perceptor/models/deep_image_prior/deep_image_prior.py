"""
https://github.com/crowsonkb/deep-image-prior
https://github.com/DmitryUlyanov/deep-image-prior
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import module_device, Tensor

from .get_hq_skip_net import get_hq_skip_net, get_offset_params, get_non_offset_params

DEFAULT_SIZE = 256
DEFAULT_SHAPE = (128, DEFAULT_SIZE, DEFAULT_SIZE)


class DeepImagePrior(nn.Module):
    def __init__(
        self,
        shape=DEFAULT_SHAPE,
        offset_type="none",
        n_scales=2,
        sigmoid=True,
        decorrelate_rgb=True,
        output_channels=3,
    ):
        """
        Represent image with latent and deep image prior network.

        Args:
            shape: (number of latent input channels, height, width)
            offset_type: type of offset to use, "1x1", "none", or "fulll"
            n_scales: number of up-downscales to use, if None, it is determined by the size of the input
            sigmoid: whether to use sigmoid activation
            decorrelate_rgb: whether to decorrelate the RGB channels
            output_channels: number of output channels
        """
        super().__init__()
        input_channels, height, width = shape
        assert height == width
        assert height % 8 == 0
        self.shape = shape
        self.n_scales = n_scales
        self.output_channels = output_channels
        self.network = get_hq_skip_net(
            input_depth=input_channels,
            offset_type=offset_type,
            num_scales=n_scales,
            decorr_rgb=decorrelate_rgb,
            sigmoid=sigmoid,
            n_channels=output_channels,
        )

    @property
    def device(self):
        return module_device(self)

    @property
    def input_channels(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def width(self):
        return self.shape[2]

    def forward(self, latents: Tensor.dims("NCHW")) -> Tensor.dims("NDHW"):
        return self.network(latents)

    def random_latents(self, size=1, n_channels=None):
        if n_channels is None:
            n_channels = self.input_channels
        return 0.1 * torch.randn(
            (size, n_channels, self.height, self.width), device=self.device
        )

    def fourier_latents(
        self,
        size=1,
        n_channels=None,
        min_log2_frequency=0.0,
        max_log2_frequency=9.0,
        log2_space=False,
    ):
        if n_channels is None:
            n_channels = self.input_channels
        assert n_channels % 4 == 0

        xs = torch.linspace(-1, 1, self.shape[2])
        ys = torch.linspace(-1, 1, self.shape[1])

        meshgrid = torch.stack(torch.meshgrid(xs, ys), dim=0)

        if log2_space:
            frequencies = 2.0 ** torch.linspace(
                min_log2_frequency, max_log2_frequency, steps=n_channels // 4
            )
        else:
            frequencies = torch.linspace(
                2.0**min_log2_frequency,
                2.0**max_log2_frequency,
                steps=n_channels // 4,
            )

        fourier_latents = torch.cat(
            [
                torch.sin(
                    meshgrid[None] * frequencies[:, None, None, None] * 2 * np.pi
                ),
                torch.cos(
                    meshgrid[None] * frequencies[:, None, None, None] * 2 * np.pi
                ),
            ],
            dim=0,
        ).flatten(end_dim=1)[None]
        return fourier_latents.to(self.device).repeat(size, 1, 1, 1).mul(0.3)

    def noisy_image_latents(self, images, n_channels=None, log_snr=-1.0):
        if n_channels is None:
            n_channels = self.input_channels

        sigma = 1 / (torch.as_tensor(log_snr).exp().sqrt() + 1)
        channels = images.shape[1]
        repeated_images = torch.stack(
            [images[:, index % channels] for index in range(n_channels)], dim=1
        )
        return 0.1 * (
            repeated_images.mul(2).sub(1) * (1 - sigma)
            + torch.randn_like(repeated_images) * sigma
        )

    def offset_parameters(self):
        return get_offset_params(self.network)

    def non_offset_parameters(self):
        return get_non_offset_params(self.network)

    def parameter_dicts(self, learning_rate):
        return [
            dict(
                params=self.offset_parameters(),
                lr=learning_rate * 0.1,
            ),
            dict(
                params=self.non_offset_parameters(),
                lr=learning_rate,
            ),
        ]


def test_deep_image_prior():
    model = DeepImagePrior().cuda()
    model(model.random_latents())
