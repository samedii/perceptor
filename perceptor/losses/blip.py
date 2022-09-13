import torch
import torch.nn.functional as F
from torchvision import transforms
from basicsr.utils.download_util import load_file_from_url

from perceptor import models
from perceptor.losses.interface import LossInterface


class BLIP(LossInterface):
    def __init__(self, name="model_base_retrieval_flickr"):
        """
        Args:
            name (str): name of the blip model. Available models are:
                - model_base_retrieval_coco
                - model_large_retrieval_coco
                - model_base_retrieval_flickr
                - model_large_retrieval_flickr
                - model_large
                - model*_base
                - model_base
                - model_base_capfilt_large
        """
        super().__init__()
        self.name = name
        self.model = models.BLIP(name)

        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def add_texts_(self, texts, weights=None):
        text_encodings = self.model.encode_texts(texts)
        return self.add_encodings_(text_encodings, weights)

    def add_images_(self, images, weights=None):
        image_encodings = self.model.encode_images(images)
        return self.add_encodings_(image_encodings, weights)

    def add_encodings_(self, encodings, weights=None):
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
        return (
            self.model.image_text_contrastive_spherical_distance(
                self.model.encode_images(images), self.encodings
            )
            * self.weights[:, None]
        ).mean()


def test_blip_loss():
    loss = (
        BLIP().add_texts_(["hello", "world"]).add_images_(torch.randn(1, 3, 256, 256))
    )
    loss(torch.randn(1, 3, 256, 256))
