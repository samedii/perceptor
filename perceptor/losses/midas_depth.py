import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from perceptor.models.midas_depth import (
    MidasDepth as MidasDepthModel,
)


class MidasDepth(LossInterface):
    def __init__(self, name="dpt_large"):
        super().__init__()
        self.model = MidasDepthModel(name=name)

    def forward(self, images):
        with torch.no_grad():
            downsampled_size = torch.tensor(images.shape[-2:]) // 4
            downsampled = F.interpolate(
                images,
                size=downsampled_size.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            upsampled = self.transform(downsampled)
            if upsampled.shape != images.shape:
                upsampled = F.interpolate(
                    upsampled,
                    size=tuple(images.shape[:2]),
                    mode="bilinear",
                    align_corners=False,
                )
        return torch.norm(images - upsampled, p=2, dim=1).mean() * 0.001
