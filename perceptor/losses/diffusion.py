import torch
import torch.nn.functional as F

from .interface import LossInterface
from perceptor import models, transforms


class Diffusion(LossInterface):
    def __init__(self, name="yfcc_2", size=None):
        super().__init__()
        self.model = models.VelocityDiffusion(name)

    def forward(self, images, t=0.7):
        if images.shape[-3:] != self.model.shape:
            images = transforms.resize(images, out_shape=self.model.shape[-2:])
        with torch.no_grad():
            denoised = self.model.denoise(self.model.diffuse(images, t), t)
        return (denoised - images).square().mean()


def test_diffusion_loss():
    model = models.VelocityDiffusion().cuda()
    loss = Diffusion("yfcc_2").cuda()
    images = torch.zeros((1, *model.shape)).cuda()
    loss(images)
