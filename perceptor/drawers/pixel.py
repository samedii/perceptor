from torch import nn
import torch.nn.functional as F

from .interface import DrawingInterface


class Pixel(DrawingInterface):
    def __init__(self, init_images):
        super().__init__()
        self.images = nn.Parameter(init_images)

    def synthesize(self, _=None):
        return self.images

    def encode(self, images, mode="bilinear"):
        return F.interpolate(
            images,
            size=tuple(self.images.shape[-2:]),
            mode=mode,
            align_corners=False,
        )

    def replace_(self, images):
        self.images.data.copy_(images.data)
        return self
