from perceptor.transforms.interface import TransformInterface


class Diffusion(TransformInterface):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, images, t, noise=None):
        return self.diffuse(images, t, noise)

    def decode(self, diffused_images, t):
        return self.predict_denoised(diffused_images, t)

    def diffuse(self, images, t, noise=None):
        return self.model.diffuse(images.mul(2).sub(1), t, noise).add(1).div(2)

    def predict_denoised(self, diffused_images, t):
        return (
            self.model.predict_denoised(diffused_images.mul(2).sub(1), t).add(1).div(2)
        )

    def velocity(self, diffused_images, t):
        return self.model(diffused_images.mul(2).sub(1), t)
