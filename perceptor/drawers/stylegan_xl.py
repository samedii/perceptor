from torch import nn

from perceptor.drawers.interface import DrawingInterface
from perceptor import models


class StyleGANXL(DrawingInterface):
    def __init__(self, n_images=1, name="imagenet128"):
        super().__init__()
        self.model = models.StyleGANXL(name)
        self.latents = nn.Parameter(self.model.latents(n_images), requires_grad=True)

    def synthesize(self, _=None):
        return self.decode(self.latents)

    def encode(self, images):
        raise NotImplementedError()

    def decode(self, latents):
        return self.model(latents)
