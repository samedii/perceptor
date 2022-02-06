from .interface import LossInterface


class Smoothness(LossInterface):
    def forward(self, images):
        images = images.contiguous().clone()
        gradient_height = images[:, :, 1:, :] - images[:, :, :-1, :]
        gradient_width = images[:, :, :, 1:] - images[:, :, :, :-1]
        sharpness = gradient_height.square().mean() + gradient_width.square().mean()
        return sharpness
