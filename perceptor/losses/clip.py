import torch

from perceptor import models
from perceptor.losses.interface import LossInterface


class CLIP(LossInterface):
    def __init__(self, name="ViT-B/16"):
        super().__init__()
        self.name = name
        self.model = models.CLIP(name)
        self.encodings = None

    def add_texts_(self, texts):
        return self.add_encodings_(self.model.encode_texts(texts))

    def add_images_(self, images):
        return self.add_encodings_(self.model.encode_images(images))

    def add_encodings_(self, encodings):
        if self.encodings is None:
            self.encodings = torch.nn.Parameter(encodings, requires_grad=False)
        else:
            self.encodings = torch.nn.Parameter(
                torch.cat([self.encodings, encodings]), requires_grad=False
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
        return spherical_distance.mean()
