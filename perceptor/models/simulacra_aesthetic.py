"""
https://github.com/crowsonkb/simulacra-aesthetic-models/blob/master/LICENSE
"""

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.download_util import load_file_from_url

from perceptor import models

CHECKPOINT_URLS = {
    "ViT-B/32": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_b_32_linear.pth",
    "ViT-B/16": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_b_16_linear.pth",
    "ViT-L/14": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_l_14_linear.pth",
}


class SimulacraAesthetic(nn.Module):
    def __init__(self, model_name="ViT-L/14"):
        """
        Simulacra aesthetic loss based on clip linear regression probe that predicts the aesthetic rating of an image.

        Args:
            model_name (str): Name of CLIP model. One of "ViT-B/16", "ViT-B/32", or "ViT-L/14".
            aesthetic_target (int): Target asthetic rating of the image (1-10).
        """
        super().__init__()

        clip_model = models.CLIP(model_name)

        checkpoint_path = load_file_from_url(
            CHECKPOINT_URLS[model_name],
            "models",
        )
        self.linear = nn.Linear(clip_model.output_channels, 1)
        self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.linear.eval()
        self.linear.requires_grad_(False)

        self.clip_model = clip_model

    def forward(self, images):
        encodings = self.clip_model.encode_images(images)
        return self.linear(F.normalize(encodings, dim=-1) * encodings.shape[-1] ** 0.5)


def test_simulacra_aesthetic():
    model = SimulacraAesthetic().cuda()
    model(torch.randn(1, 3, 256, 256).cuda())
