from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.download_util import load_file_from_url

from perceptor.drawers.interface import DrawingInterface
from perceptor import models


class BruteDiffusion(DrawingInterface):
    def __init__(self, images, noise, name="yfcc_2"):
        super().__init__()
        self.model = models.VelocityDiffusion(name)
        self.noise = noise
        self.diffused_images = nn.Parameter(
            self.encode(images),
            requires_grad=True,
        )

    def synthesize(self, _=None):
        return self.model.predict_denoised(self.diffused_images, noise=self.noise)

    def encode(self, images):
        return self.model.diffuse(images, from_noise=0, to_noise=self.noise)

    def replace_(self, diffused_images):
        self.diffused_images.data.copy_(diffused_images)
        return self
