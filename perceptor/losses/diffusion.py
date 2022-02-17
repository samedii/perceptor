import torch

from .interface import LossInterface
from perceptor import models, utils


class Diffusion(LossInterface):
    def __init__(self, name="yfcc_2"):
        super().__init__()
        self.model = models.VelocityDiffusion(name)

    def forward(self, images, t=0.7):
        with torch.no_grad():
            denoised = self.model.predict_denoised(
                self.model.diffuse(images.mul(2).sub(1), t), t
            )
            # try doing K diffusion steps?
        # utils.pil_image(denoised.detach().add(1).div(2)).save("denoised.png")
        return (denoised - images.mul(2).sub(1)).square().mean()
