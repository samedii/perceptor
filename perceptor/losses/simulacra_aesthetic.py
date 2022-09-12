import torch
from torch.nn import functional as F

from .interface import LossInterface
from perceptor import models


class SimulacraAesthetic(LossInterface):
    def __init__(self, model_name="ViT-L-14", aesthetic_target=10):
        """
        Simulacra aesthetic loss based on clip linear regression probe that predicts the aesthetic rating of an image.

        Args:
            model_name (str): Name of CLIP model. Available models are:
                - ViT-B-32
                - ViT-B-16
                - ViT-L-14
                - RN50
                - RN101
                - RN50x4
                - RN50x16
                - RN50x64
                - ViT-L-14-336
            aesthetic_target (int): Target asthetic rating of the image (1-10).
        """
        super().__init__()
        self.aesthetic_target = torch.nn.Parameter(
            torch.as_tensor(aesthetic_target).float(), requires_grad=False
        )
        self.model = models.SimulacraAesthetic(model_name)
        if model_name in ("ViT-L-14", "ViT-L-14-336"):
            self.multiplier = 0.00001
        else:
            self.multiplier = 0.001

    def forward(self, images):
        predicted_aesthetic_rating = self.model(images)
        return self.multiplier * F.mse_loss(
            predicted_aesthetic_rating,
            self.aesthetic_target.view(-1, 1),
        )


def test_simulacra_aesthetic_loss():
    loss = SimulacraAesthetic("ViT-B-32").cuda()
    loss(torch.randn(1, 3, 256, 256).cuda())
