import torch
import torch.nn.functional as F

from .interface import LossInterface
from perceptor import models, utils


class Diffusion(LossInterface):
    def __init__(self, name="yfcc_2", size=None):
        super().__init__()
        self.model = models.VelocityDiffusion(name)
        if size is not None:
            self.size = size
        elif name in ("yfcc_1", "yfcc_2"):
            self.size = 512
        else:
            self.size = 256

    def forward(self, images, t=0.7):
        # TODO: resize
        if images.shape[-2:] != (self.size, self.size):
            images = F.interpolate(images, size=(self.size, self.size), mode="bilinear")
        with torch.no_grad():
            denoised = self.model.predict_denoised(
                self.model.diffuse(images.mul(2).sub(1), t), t
            )
            # try doing K diffusion steps?
        # utils.pil_image(denoised.detach().add(1).div(2)).save("denoised.png")
        return (denoised - images.mul(2).sub(1)).square().mean()
