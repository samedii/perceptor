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


@cache
class VelocityDiffusion(torch.nn.Module):
    def __init__(self, name="yfcc_1"):
        super().__init__()
        self.name = name
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
        if x.shape[1:] != self.model.shape:
            raise ValueError(
                f"Velocity diffusion model {self.name} only works well with shape {self.model.shape}"
            )
        if hasattr(self.model, "clip_model"):
            model_fn = partial(self.model, clip_embed=self.encodings.mean(dim=0)[None])
        else:
            model_fn = self.model

        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)

        velocity = model_fn(x, t)
        return velocity.float()

    def predict_denoised(self, x, t):
        """Predict the denoised images `pred` (range -1 to 1)"""
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)
        velocity = self.forward(x, t)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return x * alphas[:, None, None, None] - velocity * sigmas[:, None, None, None]

    @staticmethod
    def diffuse(pred, t, noise=None):
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((pred.shape[0],), t).to(pred)
        if noise is None:
            noise = torch.randn_like(pred)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return pred * alphas + noise * sigmas

    @staticmethod
    def x(pred, noise, t):
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((pred.shape[0],), t).to(pred)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return pred * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]

    def noise(self, x, t, pred=None):
        """Also called eps"""
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)
        if pred is None:
            pred = self.predict_denoised(x, t)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return (x - pred * alphas[:, None, None, None]) / sigmas[:, None, None, None]
