import torch
import torch.nn.functional as F
from torchvision import transforms
from basicsr.utils.download_util import load_file_from_url

from perceptor import models
from perceptor.losses.interface import LossInterface


class BLIP(LossInterface):
    def __init__(self, name="model_base_retrieval_flickr"):
        super().__init__()
        self.name = name
        self.model = models.BLIP(name)

        self.tokenized_texts = None
        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def add_texts_(self, texts, weights=None):
        tokenized_texts, text_encodings_itc = self.model.encode_texts(texts)
        if self.tokenized_texts is None:
            self.tokenized_texts = tokenized_texts
        else:
            raise ValueError("Adding more texts is not supported")

        return self.add_encodings_(text_encodings_itc, weights)

    def add_images_(self, images, weights=None):
        _, image_encodings_itc = self.model.encode_images(images)
        return self.add_encodings_(image_encodings_itc, weights)

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
        image_embeddings, image_encodings_itc = self.model.encode_images(images)
        return (
            self.model.image_text_contrastive_spherical_distance(
                image_encodings_itc, self.encodings
            )
            * self.weights[:, None]
        ).mean() * 0.9 + (
            self.model.image_text_retrieval_probabilities(
                self.tokenized_texts, image_embeddings
            )
            * self.weights[:, None]
        ).mean() * 0.1
