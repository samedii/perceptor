from functools import partial
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from perceptor import models
from .models import get_model
from .model_urls import model_urls
import perceptor.models.velocity_diffusion.utils as utils
import perceptor.models.velocity_diffusion.sampling as sampling


@cache
class VelocityDiffusion(torch.nn.Module):
    def __init__(self, name="yfcc_2"):
        super().__init__()
        self.model = get_model(name)()
        checkpoint_path = load_file_from_url(model_urls[name], "models")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint {checkpoint_path}")
        self.model.eval()
        self.model.requires_grad_(False)
        self.encodings = None

    def to(self, device):
        super().to(device)
        if device == torch.device("cuda"):
            self.model.half()
        return self

    def add_texts_(self, texts):
        return self.add_encodings_(
            models.CLIP(self.model.clip_model).encode_texts(texts)
        )

    def add_images_(self, images):
        return self.add_encodings_(
            models.CLIP(self.model.clip_model).encode_images(images)
        )

    def add_encodings_(self, encodings):
        if self.encodings is None:
            self.encodings = torch.nn.Parameter(encodings, requires_grad=False)
        else:
            self.encodings = torch.nn.Parameter(
                torch.cat([self.encodings, encodings]), requires_grad=False
            )
        return self

    @torch.cuda.amp.autocast()
    def forward(self, x, t):
        if hasattr(self.model, "clip_model"):
            model_fn = partial(self.model, clip_embed=self.encodings)
        else:
            model_fn = self.model

        velocity = model_fn(x, t)
        return velocity.float()

    def predict_denoised(self, x, t):
        """Predict the denoised images."""
        velocity = self.forward(x, t)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        pred = x * alphas[:, None, None, None] - velocity * sigmas[:, None, None, None]
        return (pred + 1) / 2
