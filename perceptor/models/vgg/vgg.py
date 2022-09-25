import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):
    def __init__(self, name="squeeze", linear_layers=True, spatial=False):
        super().__init__()
        self.model = models.vgg19(pretrained=True).features
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images):
        """
        Args:
            images_a: images of shape (batch_size, 3, height, width) between 0 and 1
            images_b: images of shape (batch_size, 3, height, width) between 0 and 1
        """

        _, _, height, width = images.shape
        if width % 8 != 0:
            raise ValueError("Width must be divisible by 8")
        if height % 8 != 0:
            raise ValueError("Height must be divisible by 8")

        return self.model(images)


def test_vgg19():
    import torch

    model = VGG19()
    images_a = torch.randn((1, 3, 256, 256))
    model(images_a)
