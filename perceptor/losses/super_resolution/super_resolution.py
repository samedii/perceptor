import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from perceptor.transforms.super_resolution import (
    SuperResolution as SuperResolutionTransform,
)


superresolution_checkpoint_table = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


class SuperResolution(LossInterface):
    def __init__(self, name="RealESRGAN_x4plus"):
        super().__init__()
        self.transform = SuperResolutionTransform(name=name)

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
