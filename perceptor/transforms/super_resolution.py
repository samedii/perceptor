import torch
import torch.nn.functional as F

from perceptor.transforms.interface import TransformInterface
from perceptor import models


class SuperResolution(TransformInterface):
    def __init__(self, name="RealESRGAN_x2plus"):
        super().__init__()
        self.name = name
        self.model = models.SuperResolution(name)

    def encode(self, images):
        return self.model.upsampler.enhance(
            images,
            outscale=self.model.scale,
        )

    def decode(self, upsampled_images, size=None):
        if size is None:
            size = (
                torch.tensor(upsampled_images.shape[-2:]) // self.model.scale
            ).tolist()
        return F.interpolate(
            upsampled_images,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
