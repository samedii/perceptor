from torch import nn
import torch.nn.functional as F

from .interface import DrawingInterface


class Pixel(DrawingInterface):
    def __init__(self, init_image):
        super().__init__()
        self.image = nn.Parameter(init_image)

    def synthesize(self, _=None):
        return self.image

    def decode(self, image, mode="bilinear"):
        return F.interpolate(
            image,
            size=tuple(self.image.shape[-2:]),
            mode=mode,
            align_corners=False,
        )

    def replace_(self, image):
        self.image.data.copy_(image.data)
        return self
