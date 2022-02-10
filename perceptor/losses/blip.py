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

    def add_texts_(self, texts):
        tokenized_texts, text_encodings_itc = self.model.encode_texts(texts)
        if self.tokenized_texts is None:
            self.tokenized_texts = tokenized_texts
        else:
            raise ValueError("Adding more texts is not supported")

        return self.add_encodings_(text_encodings_itc)

    def add_images_(self, images):
        _, image_encodings_itc = self.model.encode_images(images)
        return self.add_encodings_(image_encodings_itc)

    def add_encodings_(self, encodings):
        if self.encodings is None:
            self.encodings = torch.nn.Parameter(encodings, requires_grad=False)
        else:
            self.encodings = torch.nn.Parameter(
                torch.cat([self.encodings, encodings]), requires_grad=False
            )
        return self

    def forward(self, images):
        image_embeddings, image_encodings_itc = self.model.encode_images(images)
        return (
            self.model.image_text_captioning_spherical_distance(
                image_encodings_itc, self.encodings
            ).mean()
            * 0.9
            + self.model.image_text_retrieval_probabilities(
                self.tokenized_texts, image_embeddings
            ).mean()
            * 0.1
        )
