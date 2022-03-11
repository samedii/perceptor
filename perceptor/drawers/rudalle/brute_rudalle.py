from torch import nn

from perceptor.drawers.interface import DrawingInterface
from .get_vae import get_vae


class BruteRuDalle(DrawingInterface):
    def __init__(self, init_images):
        super().__init__()
        self.model = (
            get_vae(dwt=False, cache_dir="models").model.eval().requires_grad_(False)
        )
        self.latent = nn.Parameter(self.encode(init_images), requires_grad=True)

    def synthesize(self, _=None):
        return self.decode(self.latent)

    def encode(self, image):
        quant, embedding_loss, info = self.model.encode(image.mul(2).sub(1))
        return quant

    def decode(self, latent):
        return self.model.decode(latent).add(1).div(2)
