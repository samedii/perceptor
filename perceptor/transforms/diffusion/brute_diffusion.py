from perceptor.transforms.interface import TransformInterface


class BruteDiffusion(TransformInterface):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, images, t, noise=None):
        return self.diffuse(images, t, noise)

    def decode(self, x, t):
        return self.predict_denoised(x, t)

    def diffuse(self, images, t, noise=None):
        return self.model.diffuse(images.mul(2).sub(1), t, noise)

    def predict_denoised(self, x, t):
        return self.model.predict_denoised(x, t).add(1).div(2)

    def velocity(self, x, t):
        return self.model(x, t)

    def x(self, predicted_images, eps, t):
        pred = predicted_images.mul(2).sub(1)
        return self.model.x(pred, eps, t)
