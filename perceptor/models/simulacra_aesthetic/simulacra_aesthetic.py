"""
https://github.com/crowsonkb/simulacra-aesthetic-models/blob/master/LICENSE
"""

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.download_util import load_file_from_url

from perceptor import models

CHECKPOINT_URLS = {
    "ViT-B-32": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_b_32_linear.pth",
    "ViT-B-16": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_b_16_linear.pth",
    "ViT-L-14": "https://raw.githubusercontent.com/crowsonkb/simulacra-aesthetic-models/master/models/sac_public_2022_06_29_vit_l_14_linear.pth",
    "RN50": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/RN50.pth",
    "RN101": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/RN101.pth",
    "RN50x4": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/RN50x4.pth",
    "RN50x16": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/RN50x16.pth",
    "RN50x64": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/RN50x64.pth",
    "ViT-L-14-336": "https://raw.githubusercontent.com/samedii/perceptor/master/perceptor/models/simulacra_aesthetic/weights/ViT-L-14-336px.pth",
}


class SimulacraAesthetic(nn.Module):
    def __init__(self, model_name="ViT-B-32"):
        """
        Simulacra aesthetic loss based on clip linear regression probe that predicts the aesthetic rating of an image.

        Args:
            model_name (str): Name of CLIP model. Available models:
                - ViT-B-32
                - ViT-B-16
                - ViT-L-14
                - RN50
                - RN101
                - RN50x4
                - RN50x16
                - RN50x64
                - ViT-L-14-336
        """
        super().__init__()

        clip_model = models.CLIP(model_name)

        checkpoint_path = load_file_from_url(
            CHECKPOINT_URLS[model_name],
            "models",
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.linear = nn.Linear(state_dict["linear.weight"].shape[1], 1)
        self.load_state_dict(state_dict)
        self.linear.eval()
        self.linear.requires_grad_(False)

        self.clip_model = clip_model

    def forward(self, images):
        encodings = self.clip_model.encode_images(images)
        return self.linear(F.normalize(encodings, dim=-1) * encodings.shape[-1] ** 0.5)


def test_simulacra_aesthetic():
    model = SimulacraAesthetic().cuda()
    model(torch.randn(1, 3, 256, 256).cuda())
