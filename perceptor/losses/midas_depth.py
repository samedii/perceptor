import torch.nn.functional as F

from perceptor.transforms.resize import resize
from perceptor.losses.interface import LossInterface
from perceptor.models.midas_depth import (
    MidasDepth as MidasDepthModel,
)


class MidasDepth(LossInterface):
    def __init__(self, name="dpt_large"):
        super().__init__()
        self.model = MidasDepthModel(name=name)

    def forward(self, images, depth_maps):
        mask = depth_maps != -1
        predicted_depth_maps = (
            resize(
                self.model(images)[:, None],
                out_shape=depth_maps.shape[-2:],
            )[:, 0]
            * mask
        )
        return (
            (normalize(predicted_depth_maps[mask]) - normalize(depth_maps[mask]))
            .square()
            .mean()
        )


def normalize(x):
    return (x - x.mean()) / x.std()


def test_midas_depth_loss():
    import torch

    loss = MidasDepth().cuda()
    images = torch.zeros((1, 3, 256, 256)).cuda()
    depth_maps = torch.zeros((1, 3, 256, 256)).cuda()
    loss(images, depth_maps)
