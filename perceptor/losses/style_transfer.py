import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from .interface import LossInterface


class StyleTransfer(LossInterface):
    def __init__(self, style_image):
        super().__init__()
        self.model = torchvision.models.vgg19(pretrained=True).features
        self.model.eval()
        self.model.requires_grad_(False)
        self.vgg_Y = nn.ParameterList(
            [
                nn.Parameter(tensor, requires_grad=False)
                for tensor in get_vgg_activations(
                    self.model,
                    [F.interpolate(style_image, size=(256, 256), mode="bilinear")],
                )
            ]
        )

    def forward(self, images):
        if images.shape[-2:] != (256, 256):
            images = F.interpolate(images, size=(256, 256), mode="bilinear")
        vgg_P = get_vgg_activations(self.model, [images])

        vgg_loss = [F.l1_loss(self.vgg_Y[i], vgg_P[i]) for i in range(len(self.vgg_Y))]
        vgg_loss_gram = [
            F.l1_loss(gram_matrix(self.vgg_Y[i]), gram_matrix(vgg_P[i]))
            for i in range(len(self.vgg_Y))
        ]

        vgg_loss = 5 * vgg_loss[2] + 15 * vgg_loss[3] + 2 * vgg_loss[4]
        vgg_loss_gram = (
            5**2 * 5e3 * vgg_loss_gram[2]
            + 15**2 * 5e3 * vgg_loss_gram[3]
            + 2**2 * 5e3 * vgg_loss_gram[4]
        )

        return (vgg_loss + vgg_loss_gram) * 0.001


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
