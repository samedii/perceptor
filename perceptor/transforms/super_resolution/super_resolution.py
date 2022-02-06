from pathlib import Path
import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from perceptor.transforms.interface import TransformInterface
from .real_esrganer import RealESRGANer


checkpoints = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


class SuperResolution(TransformInterface):
    def __init__(self, *_, name="RealESRGAN_x4plus"):
        super().__init__()
        self.name = name

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.model.requires_grad_(False)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=checkpoint_path,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )

    def encode(self, image):
        return self.upsampler.enhance(
            image,
            outscale=4,
        )

    def decode(self, upsampled_image, size=None):
        if size is None:
            size = (torch.tensor(upsampled_image.shape[-2:]) // 4).tolist()
        return F.interpolate(
            upsampled_image,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
