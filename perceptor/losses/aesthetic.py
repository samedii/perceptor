import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.download_util import load_file_from_url

from .interface import LossInterface
from perceptor import models


class Aesthetic(LossInterface):
    def __init__(self, aesthetic_rating=1.0):
        """
        Aesthetic loss based on a classifier that predicts the aesthetic rating of an image.

        Args:
            aesthetic_rating (float): Target asthetic rating of the image (0-1).
        """
        super().__init__()
        self.aesthetic_rating = aesthetic_rating

        self.model = models.CLIP("ViT-B/16")

        checkpoint_path = load_file_from_url(
            "https://dazhi.art/f/ava_vit_b_16_linear.pth", "models"
        )
        layer_weights = torch.load(checkpoint_path)
        self.aesthetic_head = nn.Linear(512, 1)
        self.aesthetic_head.load_state_dict(layer_weights)
        self.aesthetic_head.eval()
        self.aesthetic_head.requires_grad_(False)

    def forward(self, images):
        aes_rating = self.aesthetic_head(self.model.encode_images(images))
        return (aes_rating - self.aesthetic_rating * 10).square().mean() * 0.02
