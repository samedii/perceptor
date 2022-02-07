from torch import nn
import torch.nn.functional as F

from perceptor.drawers.interface import DrawingInterface
from .compression import compress_jpeg
from .decompression import decompress_jpeg


class JPEG(DrawingInterface):
    def __init__(self, init_images, factor=1):
        super().__init__()
        self.shape = init_images.shape
        height, width, *_ = init_images.shape[-2:]
        self.compress_jpeg = compress_jpeg(factor=factor)
        self.decompress_jpeg = decompress_jpeg(height, width, factor=factor)
        self.ycbcr = nn.ParameterList(
            [nn.Parameter(parameter) for parameter in self.encode(init_images)]
        )

    def synthesize(self, _=None):
        return self.decode(self.ycbcr)

    def encode(self, image):
        return self.compress_jpeg(
            F.interpolate(image, size=self.shape[-2:], mode="bilinear")
        )

    def decode(self, ycbcr):
        return self.decompress_jpeg(*ycbcr)
