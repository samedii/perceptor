from .interface import LossInterface


class SphericalDistance(LossInterface):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images_a, images_b):
        spherical_distance = (
            (
                self.model.encode_images(images_a)[:, None]
                - self.model.encode_images(images_b)[None, :]
            )
            .norm(dim=2)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )
        return spherical_distance.mean()
