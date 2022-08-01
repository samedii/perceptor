import lpips

from .interface import LossInterface


class LPIPS(LossInterface):
    def __init__(self, name="squeeze", linear_layers=True, spatial=False):
        """
        LPIPS loss. Expects images of shape (batch_size, 3, height, width) between 0 and 1.

        Args:
            name (str): name of the loss. Available options: ["alex", "vgg", "squeeze"]
        """
        super().__init__()
        self.model = lpips.LPIPS(
            net=name, lpips=linear_layers, spatial=spatial, verbose=False
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images_a, images_b):
        """
        Args:
            images_a: images of shape (batch_size, 3, height, width) between 0 and 1
            images_b: images of shape (batch_size, 3, height, width) between 0 and 1
        """
        return self.model(images_a, images_b, normalize=True)


def test_lpips():
    import torch

    model = LPIPS()
    images_a = torch.randn((1, 3, 256, 256))
    images_b = torch.randn((1, 3, 256, 256))
    model(images_a, images_b)
