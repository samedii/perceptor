from pathlib import Path
import torch
from lantern import module_device
import lit

from perceptor.transforms import resize
from perceptor.losses.interface import LossInterface


class LiT(LossInterface):
    def __init__(
        self,
        name="LiT-L16L",
        cache_dir=Path("models"),
    ):
        """
        LiT image-text similarity

        Args:
            model_name (str): Name of the model to load. "LiT-B16B_2" or "LiT-L16L"
            cache_dir (str): Path to the directory where the model is cached.
        """
        super().__init__()
        self.model = lit.LiT(name, cache_dir)
        self.encodings = None
        self.weights = None

    @property
    def device(self):
        return module_device(self)

    def add_texts_(self, texts, weights=None):
        return self.add_encodings_(self.model.encode_texts(texts), weights)

    def add_images_(self, images, weights=None):
        if images.shape[-2:] != self.model.image_size:
            images = resize(images, out_shape=self.model.image_size)
        return self.add_encodings_(self.model.encode_images(images), weights)

    def add_encodings_(
        self,
        encodings,
        weights=None,
    ):
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
        if images.shape[-2:] != self.model.image_size:
            images = resize(images, out_shape=self.model.image_size)
        image_encodings = self.model.encode_images(images)
        spherical_distance = (
            (image_encodings[:, None] - self.encodings[None, :])
            .norm(dim=2)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )
        return (spherical_distance * self.weights).mean()


def test_lit():
    torch.set_grad_enabled(False)
    loss = LiT().add_texts_(["hello", "world"]).add_images_(torch.randn(1, 3, 256, 256))

    image = torch.randn(1, 3, 256, 256).requires_grad_()
    with torch.enable_grad():
        loss(image).backward()

    assert image.grad is not None
