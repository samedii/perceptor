from torchvision import transforms
from resmem import ResMem

from .interface import LossInterface


class Memorability(LossInterface):
    def __init__(self):
        super().__init__()
        self.recenter = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(227),
            ]
        )
        self.model = ResMem(pretrained=True)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images):
        image_x = self.recenter(images)
        prediction = self.model(image_x)
        return prediction.mean() * 0.05
