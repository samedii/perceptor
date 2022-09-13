import torch

from perceptor import models
from perceptor.losses.interface import LossInterface


class CLOOB(LossInterface):
    def __init__(self, name="16-epochs"):
        """
        Args:
            name: name of the cloob model. Available models are:
                - 16-epochs
                - 32-epochs
        """
        super().__init__()
        self.name = name
        self.model = models.CLOOB(name)
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


def test_cloob_loss():
    loss = (
        CLOOB().add_texts_(["hello", "world"]).add_images_(torch.randn(1, 3, 256, 256))
    )
    loss(torch.randn(1, 3, 256, 256))
