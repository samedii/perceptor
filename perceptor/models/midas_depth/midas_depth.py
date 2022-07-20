"""
MiDaS model from https://github.com/isl-org/MiDaS and
inference impementation from https://github.com/alembics/disco-diffusion/blob/main/disco.py
"""
from multiprocessing.sharedctypes import Value
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from basicsr.utils.download_util import load_file_from_url

from perceptor import utils, transforms
from .dpt_depth import DPTDepthModel
from .midas_net import MidasNet
from .midas_net_custom import MidasNet_small

# from .transforms import Resize, NormalizeImage, PrepareForNet

checkpoints = {
    "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
    "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_nyu": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt",
    "dpt_hybrid_kitti": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt",
}


@utils.cache
class MidasDepth(nn.Module):
    def __init__(self, name="dpt_large", optimize=True):
        """
        Args:
            name (str, optional): Name of the model. Suggested value:
                dpt_large - highest quality
                dpt_hybrid - moderately less quality, but better speed on CPU and slower GPUs
                midas_v21_small - real-time applications on resource-constrained devices
        """
        super().__init__()
        self.name = name
        self.model = None
        self.optimize = optimize

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        if self.name == "dpt_large":  # DPT-Large
            self.model = DPTDepthModel(
                path=checkpoint_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            self.resize_mode = "minimal"
            self.normalization = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.name == "dpt_hybrid":  # DPT-Hybrid
            self.model = DPTDepthModel(
                path=checkpoint_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            self.resize_mode = "minimal"
            self.normalization = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.name == "dpt_hybrid_nyu":  # DPT-Hybrid-NYU
            self.model = DPTDepthModel(
                path=checkpoint_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            self.resize_mode = "minimal"
            self.normalization = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.name == "dpt_hybrid_kitti":  # Assumed same as DPT-Hybrid-NYU
            self.model = DPTDepthModel(
                path=checkpoint_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            self.resize_mode = "minimal"
            self.normalization = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.name == "midas_v21":
            self.model = MidasNet(checkpoint_path, non_negative=True)
            self.net_w, self.net_h = 384, 384
            self.resize_mode = "upper_bound"
            self.normalization = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif self.name == "midas_v21_small":
            self.model = MidasNet_small(
                checkpoint_path,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={"expand": True},
            )
            self.net_w, self.net_h = 256, 256
            self.resize_mode = "upper_bound"
            self.normalization = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            raise ValueError(f"midas_model_type '{self.name}' not implemented")

        self.image_size = (self.net_h, self.net_w)

        self.model.eval()
        self.model.requires_grad_(False)

    def to(self, device):
        if self.optimize:
            if device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)
                self.model = self.model.half()
        return super().to(device)

    @torch.cuda.amp.autocast()
    def forward(self, images):
        if images.shape[-2:] != self.image_size:
            images = transforms.resize(
                images,
                out_shape=self.image_size,
            )
        return self.model(self.normalization(images)).float()
