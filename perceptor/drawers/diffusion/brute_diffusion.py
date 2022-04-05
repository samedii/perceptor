import torch
from torch import nn

from perceptor.drawers.interface import DrawingInterface


class BruteDiffusion(DrawingInterface):
    def __init__(self, model, diffused_images, t):
        super().__init__()
        self.model = model
        self.t = t
        self.diffused_images = nn.Parameter(
            diffused_images,
            requires_grad=True,
        )

    @staticmethod
    def from_image(model, images, t, noise=None):
        drawer = BruteDiffusion(
            model,
            torch.zeros_like(images),
            t,
        )
        return drawer.replace_(drawer.encode(images, noise))

    @property
    def x(self):
        return self.diffused_images.mul(2).sub(1)

    def synthesize(self, _=None):
        return (
            self.model.predict_denoised(self.diffused_images.mul(2).sub(1), self.t)
            .add(1)
            .div(2)
        )

    def encode(self, images, noise=None):
        return (
            self.model.diffuse(images.mul(2).sub(1), t=self.t, noise=noise)
            .add(1)
            .div(2)
        )

    def replace_(self, diffused_images):
        self.diffused_images.data.copy_(diffused_images)
        return self

    def noise(self):
        return self.model.noise(self.x, self.t)
