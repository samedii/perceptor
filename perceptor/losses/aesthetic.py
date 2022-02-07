import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.download_util import load_file_from_url

from .interface import LossInterface
from perceptor.losses.clip.clip_base import get_clip_perceptor


class Aesthetic(LossInterface):
    def __init__(self, aesthetic_rating=10):
        super().__init__()
        self.aesthetic_rating = aesthetic_rating

        self.model = get_clip_perceptor("ViT-B/16", torch.device("cuda"))
        self.model.eval()
        self.model.requires_grad_(False)

        checkpoint_path = load_file_from_url(
            "https://dazhi.art/f/ava_vit_b_16_linear.pth", "models"
        )
        layer_weights = torch.load(checkpoint_path)
        self.aesthetic_head = nn.Linear(512, 1)
        self.aesthetic_head.load_state_dict(layer_weights)
        self.aesthetic_head.eval()
        self.aesthetic_head.requires_grad_(False)

    def forward(self, images):
        image_encodings = self.model.encode_image(images)
        aes_rating = self.aesthetic_head(F.normalize(image_encodings, dim=-1))
        return (aes_rating - self.aesthetic_rating).square().mean() * 0.02
