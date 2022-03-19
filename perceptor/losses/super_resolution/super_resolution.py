import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from perceptor.transforms.super_resolution import (
    SuperResolution as SuperResolutionTransform,
)


class SuperResolution(LossInterface):
    def __init__(self, name="x4", mode="bicubic", align_corners=False):
        super().__init__()
        self.transform = SuperResolutionTransform(name=name)
        self.mode = "bicubic"
        self.align_corners = False

    def forward(self, images):
        with torch.no_grad():
            downsampled_size = torch.tensor(images.shape[-2:]) // 4
            downsampled = F.interpolate(
                images,
                size=downsampled_size.tolist(),
                mode=self.mode,
                align_corners=self.align_corners,
                # antialias=True,  # pytorch 1.11.0
            )
            upsampled = self.transform(downsampled)
            if upsampled.shape != images.shape:
                upsampled = F.interpolate(
                    upsampled,
                    size=tuple(images.shape[:2]),
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
        return torch.norm(images - upsampled, p=2, dim=1).mean() * 0.001
