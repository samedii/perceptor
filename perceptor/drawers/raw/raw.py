from torch import nn

from ..interface import DrawingInterface
from perceptor.transforms import resize
from .init.fractal import fractal
from .init.gradient import gradient


class Raw(DrawingInterface):
    def __init__(self, init_images):
        """
        Minimal container for a nn.Parameter with init helpers.

        Usage:

            images = Raw(Raw.random_fractal_image((1, 3, 256, 256)))
        """
        super().__init__()
        self.images = nn.Parameter(init_images)

    def synthesize(self, _=None):
        return self.images

    def encode(self, images, mode="bilinear"):
        return resize(
            images,
            out_shape=tuple(self.images.shape[-2:]),
            resample=mode,
        )

    def replace_(self, images):
        self.images.data.copy_(images.data)
        return self

    @staticmethod
    def random_fractal_image(shape):
        return fractal(shape)

    @staticmethod
    def random_gradient_image(shape):
        return gradient(shape)


def test_raw():
    import torch

    Raw(torch.zeros(1, 3, 128, 128))


def test_raw_fractal():
    Raw.random_fractal_image((1, 3, 256, 256))


def test_raw_gradient():
    Raw.random_gradient_image((1, 3, 256, 256))
