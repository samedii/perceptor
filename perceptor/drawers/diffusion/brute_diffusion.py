from torch import nn

from perceptor.drawers.interface import DrawingInterface
from perceptor import models


class BruteDiffusion(DrawingInterface):
    def __init__(self, images, t, name="yfcc_2"):
        super().__init__()
        self.model = models.VelocityDiffusion(name)
        self.t = t
        self.diffused_images = nn.Parameter(
            self.encode(images),
            requires_grad=True,
        )

    def synthesize(self, _=None):
        return self.model.predict_denoised(self.diffused_images, self.t)

    def encode(self, images):
        return self.model.diffuse(images, from_noise=0, to_noise=self.noise)

    def replace_(self, diffused_images):
        self.diffused_images.data.copy_(diffused_images)
        return self
