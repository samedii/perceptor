import torch
from torch.nn import functional as F

from .interface import LossInterface
from perceptor import models


class SimulacraAesthetic(LossInterface):
    def __init__(self, model_name="ViT-L/14", aesthetic_target=10):
        """
        Simulacra aesthetic loss based on clip linear regression probe that predicts the aesthetic rating of an image.

        Args:
            model_name (str): Name of CLIP model. One of "ViT-B/16", "ViT-B/32", or "ViT-L/14".
            aesthetic_target (int): Target asthetic rating of the image (1-10).
        """
        super().__init__()
        self.aesthetic_target = torch.nn.Parameter(
            torch.as_tensor(aesthetic_target).float(), requires_grad=False
        )
        self.model = models.SimulacraAesthetic(model_name)
        if model_name == "ViT-L/14":
            self.multiplier = 0.00001
        else:
            self.multiplier = 0.001

    def forward(self, images):
        predicted_aesthetic_rating = self.model(images)
        return self.multiplier * F.mse_loss(
            predicted_aesthetic_rating,
            self.aesthetic_target,
        )


def test_simulacra_aesthetic_loss():
    loss = SimulacraAesthetic().cuda()
    loss(torch.randn(1, 3, 256, 256).cuda())
