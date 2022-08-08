import torch

from perceptor import models
from perceptor.losses.interface import LossInterface


class OpenCLIP(LossInterface):
    def __init__(self, architecture="ViT-B-32", weights="laion2b_e16"):
        """
        Args:
            archicture (str): name of the clip model
            weights (str): name of the weights

            Available weight/model combinations are (in order of relevance):
            - ("ViT-B-32", "laion2b_e16") (65.62%)
            - ("ViT-B-16-plus-240", "laion400m_e32") (69.21%)
            - ("ViT-B-16", "laion400m_e32") (67.07%)
            - ("ViT-B-32", "laion400m_e32") (62.96%)
            - ("ViT-L-14", "laion400m_e32") (72.77%)
            - ("RN101", "yfcc15m") (34.8%)
            - ("RN50", "yfcc15m") (32.7%)
            - ("RN50", "cc12m") (36.45%)
            - ("RN50-quickgelu", "openai")
            - ("RN101-quickgelu", "openai")
            - ("RN50x4", "openai")
            - ("RN50x16", "openai")
            - ("RN50x64", "openai")
            - ("ViT-B-32-quickgelu", "openai")
            - ("ViT-B-16", "openai")
            - ("ViT-L-14", "openai")
            - ("ViT-L-14-336", "openai")
        """
        super().__init__()
        self.architecture = architecture
        self.weights = weights
        self.model = models.OpenCLIP(architecture, weights)
        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

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


def test_open_clip():
    loss = (
        OpenCLIP()
        .add_texts_(["hello", "world"])
        .add_images_(torch.randn(1, 3, 256, 256))
    )
    loss(torch.randn(1, 3, 256, 256))
