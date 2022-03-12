import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from huggingface_hub import hf_hub_url, cached_download

from perceptor import utils
from .real_esrganer import RealESRGANer
from .srvgg_net_compact import SRVGGNetCompact
from .custom_rrdbnet_arch import CustomRRDBNet


checkpoints = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGANv2-animevideo-xsx2": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth",
    "RealESRGANv2-animevideo-xsx4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth",
}

sberbank_checkpoints = {
    "x2": dict(
        scale=2,
        repo_id="shonenkov/rudalle-utils",
        filename="RealESRGAN_x2.pth",
    ),
    "x4": dict(
        scale=4,
        repo_id="shonenkov/rudalle-utils",
        filename="RealESRGAN_x4.pth",
    ),
    "x8": dict(
        scale=8,
        repo_id="shonenkov/rudalle-utils",
        filename="RealESRGAN_x8.pth",
    ),
}


@utils.cache
class SuperResolution(torch.nn.Module):
    def __init__(self, name="x4", half=False):
        super().__init__()
        self.name = name

        if self.name in sberbank_checkpoints:
            config = sberbank_checkpoints[name]
            self.scale = config["scale"]
            cache_dir = "models"
            config_file_url = hf_hub_url(
                repo_id=config["repo_id"], filename=config["filename"]
            )
            cached_download(
                config_file_url, cache_dir=cache_dir, force_filename=config["filename"]
            )
            model = CustomRRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.scale,
            )
            checkpoint_path = f"{cache_dir}/{config['filename']}"
        else:
            checkpoint_path = load_file_from_url(
                checkpoints[self.name],
                "models",
            )

            if self.name in ["RealESRGAN_x4plus", "RealESRNet_x4plus"]:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                self.scale = 4
            elif self.name in ["RealESRGAN_x4plus_anime_6B"]:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                )
                self.scale = 4
            elif self.name in ["RealESRGAN_x2plus"]:  # x2 RRDBNet model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                self.scale = 2
            elif self.name in [
                "RealESRGANv2-anime-xsx2",
                "RealESRGANv2-animevideo-xsx2-nousm",
                "RealESRGANv2-animevideo-xsx2",
            ]:
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=16,
                    upscale=2,
                    act_type="prelu",
                )
                self.scale = 2
            elif self.name in [
                "RealESRGANv2-anime-xsx4",
                "RealESRGANv2-animevideo-xsx4-nousm",
                "RealESRGANv2-animevideo-xsx4",
            ]:
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=16,
                    upscale=4,
                    act_type="prelu",
                )
                self.scale = 4

        model.eval()
        model.requires_grad_(False)

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=checkpoint_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

    @torch.cuda.amp.autocast()
    def forward(self, image):
        return self.upsample(image)

    def upsample(self, image):
        return self.upsampler.enhance(
            image,
            outscale=self.scale,
        ).float()

    def downsample(self, upsampled_image, size=None):
        if size is None:
            size = (torch.tensor(upsampled_image.shape[-2:]) // self.scale).tolist()
        return F.interpolate(
            upsampled_image,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
