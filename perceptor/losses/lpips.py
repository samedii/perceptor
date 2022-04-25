import lpips

from .interface import LossInterface


class LPIPS(LossInterface):
    def __init__(self, name="squeeze", linear_layers=True):
        """
        Args:
            name (str): name of the loss. Available options: ["alex", "vgg", "squeeze"]
        """
        super().__init__()
        self.model = lpips.LPIPS(net=name, lpips=linear_layers)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images_a, images_b):
        return self.model(images_a.mul(2).sub(1), images_b.mul(2).sub(1))
