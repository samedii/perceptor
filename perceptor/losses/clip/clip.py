from pathlib import Path
import json
import torch
import torch.nn.functional as F

from perceptor import models
from perceptor.losses.interface import LossInterface


class CLIP(LossInterface):
    def __init__(self, name="ViT-B-32"):
        """
        Args:
            name: name of the clip model. Available models are:
                - RN50
                - RN101
                - RN50x4
                - RN50x16
                - RN50x64
                - ViT-B-32
                - ViT-B-16
                - ViT-L-14
                - ViT-L-14-336
        """
        super().__init__()
        self.name = name
        self.model = models.CLIP(name)
        self.encodings = None
        self.weights = None
        if name in ("ViT-L-14", "ViT-L-14-336"):
            self.multiplier = 0.01
        else:
            self.multiplier = 1.0

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def mul_(self, multiplier):
        self.multiplier *= multiplier
        return self

    def add_texts_(self, texts, weights=None):
        return self.add_encodings_(self.model.encode_texts(texts), weights)

    def add_images_(self, images, weights=None):
        return self.add_encodings_(self.model.encode_images(images), weights)

    def add_text_off_(self, weight=None):
        textoff_json = json.loads(
            Path("perceptor/losses/clip/vectors/textoff.json").read_text()
        )
        if self.name in textoff_json:
            textoff = torch.tensor(textoff_json[self.name])
            return self.add_encodings_(textoff, weight)
        else:
            raise ValueError(f"There is no textoff for this model: {self.name}")

    def add_encodings_(
        self,
        encodings,
        weights=None,
    ):
        if isinstance(weights, list) or isinstance(weights, tuple):
            weights = torch.tensor(weights)
        elif weights is None:
            weights = torch.ones_like(encodings[:, 0])

        if self.encodings is None:
            self.encodings = torch.nn.Parameter(
                F.normalize(encodings).to(self.device), requires_grad=False
            )
            self.weights = torch.nn.Parameter(
                weights.to(self.device),
                requires_grad=False,
            )
        else:
            self.encodings = torch.nn.Parameter(
                torch.cat([self.encodings, F.normalize(encodings).to(self.device)]),
                requires_grad=False,
            )
            self.weights = torch.nn.Parameter(
                torch.cat([self.weights, weights.to(self.device)]),
                requires_grad=False,
            )
        return self

    def forward(self, images):
        image_encodings = self.model.encode_images(images)
        spherical_distance = (
            (image_encodings[:, None] - self.encodings[None, :])
            .norm(dim=2)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )
        return (spherical_distance * self.weights).mean().mul(self.multiplier)


def test_clip_loss():
    loss = (
        CLIP().add_texts_(["hello", "world"]).add_images_(torch.randn(1, 3, 256, 256))
    )
    loss(torch.randn(1, 3, 256, 256))
