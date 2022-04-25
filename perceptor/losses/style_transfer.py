import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from perceptor.transforms.resize import resize
from .interface import LossInterface


class StyleTransfer(LossInterface):
    def __init__(self, style_images=None):
        """Original style transfer loss"""
        super().__init__()
        self.model = torchvision.models.vgg19(pretrained=True).features
        self.model.eval()
        self.model.requires_grad_(False)

        if style_images is not None:
            self.encodings = nn.ParameterList(
                [
                    nn.Parameter(tensor, requires_grad=False)
                    for tensor in self.encode(style_images)
                ]
            )

    def encode(self, images):
        if images.shape[-2:] != (256, 256):
            images = resize(images, out_shape=(256, 256))
        return get_vgg_activations(self.model, [images])

    def loss(self, encodings_a, encodings_b):
        vgg_loss = [
            F.l1_loss(encodings_a[i], encodings_b[i]) for i in range(len(encodings_a))
        ]
        vgg_loss_gram = [
            F.l1_loss(gram_matrix(encodings_a[i]), gram_matrix(encodings_b[i]))
            for i in range(len(encodings_a))
        ]

        vgg_loss = 5 * vgg_loss[2] + 15 * vgg_loss[3] + 2 * vgg_loss[4]
        vgg_loss_gram = (
            5**2 * 5e3 * vgg_loss_gram[2]
            + 15**2 * 5e3 * vgg_loss_gram[3]
            + 2**2 * 5e3 * vgg_loss_gram[4]
        )

        return (vgg_loss + vgg_loss_gram) * 0.001

    def forward(self, images_a, images_b=None):
        if images_b is None:
            encodings_b = self.encodings
        else:
            encodings_b = self.encode(images_b)
        return self.loss(self.encode(images_a), encodings_b)


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def get_vgg_activations(vgg16, X_list):
    activations = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]
    for i, (start_index, end_index) in enumerate(activations):
        X_list.append(vgg16[start_index:end_index](X_list[i]))
    return X_list
