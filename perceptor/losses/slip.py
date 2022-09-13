import torch

from perceptor import models
from perceptor.losses.interface import LossInterface


class SLIP(LossInterface):
    def __init__(self, name="SLIP_CC12M"):
        """
        Args:
            name: name of the slip model. Available models are:
                - SLIP_VITS16
                - SLIP_VITB16
                - SLIP_VITL16
                - CLIP_VITS16
                - CLIP_VITB16
                - CLIP_VITL16
                - SLIP_CC3M
                - SLIP_CC12M
        """
        super().__init__()
        self.name = name
        self.model = models.SLIP(name)
        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def add_texts_(self, texts, weights=None):
        return self.add_encodings_(self.model.encode_texts(texts), weights)

    def add_images_(self, images, weights=None):
        return self.add_encodings_(self.model.encode_images(images), weights)

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
                encodings.to(self.device), requires_grad=False
            )
            self.weights = torch.nn.Parameter(
                weights.to(self.device),
                requires_grad=False,
            )
        else:
            self.encodings = torch.nn.Parameter(
                torch.cat([self.encodings, encodings.to(self.device)]),
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
        return (spherical_distance * self.weights).mean()


def test_slip_loss():
    loss = (
        SLIP().add_texts_(["hello", "world"]).add_images_(torch.randn(1, 3, 256, 256))
    )
    loss(torch.randn(1, 3, 256, 256))
