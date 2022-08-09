import numpy as np
from torch import nn
from basicsr.utils.download_util import load_file_from_url
from lantern import Tensor

from perceptor import utils, transforms
from .infer import InferenceHelper

MAX_ADABINS_AREA = 500000
MIN_ADABINS_AREA = 448 * 448

checkpoints = dict(
    nyu="https://s3.eu-central-1.wasabisys.com/nextml-model-data/ada-bins/AdaBins_nyu.pt",
    kitti="https://s3.eu-central-1.wasabisys.com/nextml-model-data/ada-bins/AdaBins_kitti.pt",
)


@utils.cache
class AdaBinsDepth(nn.Module):
    def __init__(self, name="nyu"):
        """
        Args:
            name (str, optional): Name of the model. Available weights: "nyu" and "kitti"
        """
        super().__init__()
        self.name = name
        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )
        self.model = InferenceHelper(name, checkpoint_path)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images: Tensor.dims("NCHW")) -> Tensor.dims("NCHW"):
        h, w = images.shape[-2:]
        image_area = w * h
        if image_area > MAX_ADABINS_AREA:
            scale = np.sqrt(MAX_ADABINS_AREA) / np.sqrt(image_area)
            images = transforms.resize(
                images, out_shape=(int(w * scale), int(h * scale)), resample="lancsoz3"
            )
        elif image_area < MIN_ADABINS_AREA:
            scale = np.sqrt(MIN_ADABINS_AREA) / np.sqrt(image_area)
            images = transforms.resize(
                images, out_shape=(int(w * scale), int(h * scale)), resample="bicubic"
            )

        return self.model.predict(images)


def test_adabins_depth():
    import requests
    from PIL import Image
    import torch
    import torchvision.transforms.functional as TF

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    images = (
        TF.to_tensor(Image.open(requests.get(image_url, stream=True).raw))[None]
        .cuda()
        .requires_grad_()
    )

    model = AdaBinsDepth().cuda()

    with torch.enable_grad():
        depths = model(images)
        depths.mean().backward()

    assert images.grad is not None

    utils.pil_image(
        torch.cat(
            [
                images,
                (depths.repeat(1, 3, 1, 1) - depths.min())
                / (depths.max() - depths.min()),
            ]
        )
    ).save("tests/adabins_depth.png")
