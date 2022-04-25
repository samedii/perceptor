from perceptor.transforms.resize import resize
from .interface import LossInterface


class Resize(LossInterface):
    def __init__(self, size=None):
        super().__init__()
        self.size = size

    def forward(self, images_a, images_b, size=None):
        if size is None:
            size = self.size

        return (
            (resize(images_a, out_shape=size) - resize(images_b, out_shape=size))
            .square()
            .mean()
        )
