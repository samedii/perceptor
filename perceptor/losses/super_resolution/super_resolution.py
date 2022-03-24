import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from perceptor import transforms


class SuperResolution(LossInterface):
    def __init__(self, name="x2", pre_downscale=None, half=True, mode="bicubic"):
        super().__init__()
        self.transform = transforms.SuperResolution(name, half)
        self.mode = mode
        self.pre_downscale = (
            self.transform.model.scale if pre_downscale is None else pre_downscale
        )

    def forward(self, images):
        with torch.no_grad():
            downsampled_size = torch.tensor(images.shape[-2:]) // self.pre_downscale
            scale = downsampled_size / torch.tensor(images.shape[-2:])
            downsampled = transforms.resize(
                images, scale.tolist(), downsampled_size, resample=self.mode
            )
            upsampled = self.transform(downsampled)
            if upsampled.shape != images.shape:
                scale = torch.tensor(images.shape[-2:]) / torch.tensor(
                    upsampled.shape[-2:]
                )
                upsampled = transforms.resize(
                    upsampled,
                    scale.tolist(),
                    out_shape=tuple(images.shape[-2:]),
                    resample=self.mode,
                )
        return (images - upsampled).square().mean()
