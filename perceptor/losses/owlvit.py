from typing import List
import torch

from perceptor.models.owlvit.owlvit import OWLViTEncodings
from perceptor import models
from perceptor.losses.interface import LossInterface


class OWLViT(LossInterface):
    def __init__(self):
        """
        OWL-ViT zero-shot text-conditioned bounding box model.

        Loss is designed in such a way that only one example of the text prompt
        is expected to be found in the image.
        """
        super().__init__()
        self.model = models.OWLViT()
        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def to(self, device):
        if self.encodings is not None:
            self.encodings.to(device)
        if self.weights is not None:
            self.weights.to(device)
        return super().to(device)

    def cuda(self):
        return self.to(torch.device("cuda"))

    def cpu(self):
        return self.to(torch.device("cpu"))

    def add_texts_(self, texts: List[str], weights=None):
        return self.add_encodings_(self.model.encode_texts([texts]), weights)

    def add_images_(self, images, weights=None):
        # return self.add_encodings_(self.model.encode_images(images), weights)
        raise NotImplementedError()

    def add_encodings_(
        self,
        encodings: OWLViTEncodings,
        weights=None,
    ):
        if isinstance(weights, list) or isinstance(weights, tuple):
            weights = torch.tensor(weights)
        elif weights is None:
            weights = torch.ones(len(encodings.texts))

        if self.encodings is None:
            self.encodings = encodings
            self.weights = torch.nn.Parameter(
                weights.to(self.device),
                requires_grad=False,
            )
        else:
            raise ValueError("OWLViT can only have one set of encodings")
        return self

    def forward(self, images, top_k=5):
        predictions = self.model(images, self.encodings)
        loss = torch.tensor(0.0, device=self.device)
        for label_index, weight in enumerate(self.weights):
            loss -= (
                predictions.logits[:, :, label_index]
                .flatten(start_dim=1)
                .log_softmax(dim=1)
                .sort(dim=1)[0][:, -top_k:]
                .mean()
                * weight
            )

        return loss * 0.01


def test_owlvit_loss():
    text_prompts = [
        "hello",
        "world",
    ]
    loss = OWLViT().add_texts_(text_prompts).cuda()
    loss(torch.zeros(3, 3, 480, 480).cuda())
