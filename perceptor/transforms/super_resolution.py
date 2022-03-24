import torch
import torch.nn.functional as F

from perceptor.transforms.interface import TransformInterface
from perceptor import models
from perceptor.transforms.resize import resize


class SuperResolution(TransformInterface):
    def __init__(self, name="x4", half=False):
        super().__init__()
        self.name = name
        self.model = models.SuperResolution(name, half)

    def encode(self, images):
        return self.model.upsample(images)

    def decode(self, upsampled_images, size=None):
        if size is None:
            size = (
                torch.tensor(upsampled_images.shape[-2:]) // self.model.scale
            ).tolist()
        return resize(
            upsampled_images,
            out_shape=size,
        )
