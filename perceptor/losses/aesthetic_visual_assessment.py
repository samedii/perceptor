import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.download_util import load_file_from_url

from .interface import LossInterface
from perceptor import models


class AestheticVisualAssessment(LossInterface):
    def __init__(self, aesthetic_target=10, mode="expected"):
        """
        Aesthetic visual assessment loss based on a classifier that predicts the aesthetic rating of an image.

        Args:
            aesthetic_target (int): Target asthetic rating of the image (1-10).
            mode (str): How to compute the loss. Available modes are:
                - "logit": compute the loss based on the logits of the classifier.
                - "expected": compute the loss based on the expected value of the classifier.
                - "probability": compute the loss based on the probability of the classifier.
        """
        super().__init__()
        self.aesthetic_target = aesthetic_target
        self.mode = mode

        self.model = models.CLIP("ViT-B-16")

        checkpoint_path = load_file_from_url(
            "http://batbot.tv/ai/models/v-diffusion/ava_vit_b_16_full.pth",
            "models",
        )
        self.aesthetic_head = nn.Linear(512, 10)
        self.aesthetic_head.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")
        )
        self.aesthetic_head.eval()
        self.aesthetic_head.requires_grad_(False)

    def forward(self, images):
        log_probs = self.aesthetic_head(self.model.encode_images(images))
        if self.mode == "logit":
            return -log_probs[..., self.aesthetic_target - 1].mean().mul(0.01)
        elif self.mode == "expected":
            expected_target = F.softmax(log_probs, dim=-1) * torch.arange(10).add(1).to(
                images.device
            )
            return (expected_target - self.aesthetic_target).square().mean().mul(0.01)
        elif self.mode == "probability":
            return -F.softmax(log_probs, dim=-1)[..., self.aesthetic_target - 1].mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def test_aesthetic_visual_assessment():
    loss = AestheticVisualAssessment().cuda()
    loss(torch.randn(1, 3, 256, 256).cuda())
