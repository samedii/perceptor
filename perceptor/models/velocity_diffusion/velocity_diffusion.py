from functools import partial
import torch
import lantern
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from perceptor import models
from .models import get_model
from .model_urls import model_urls
from . import diffusion_space, utils
from .predictions import Predictions


class Model(torch.nn.Module):
    def __init__(self, name="yfcc_2"):
        """
        Args:
            name: The name of the model. Available models are:
                - yfcc_2
                - yfcc_1
                - cc12m_1_cfg (conditioned)
                - wikiart
        """
        super().__init__()
        self.name = name
        self.model = get_model(name)()
        checkpoint_path = load_file_from_url(model_urls[name], "models")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint {checkpoint_path}")
        self.model.eval()
        self.model.requires_grad_(False)

    def to(self, device):
        super().to(device)
        if device == torch.device("cuda"):
            self.model.half()
        return self

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def shape(self):
        return self.model.shape

    @staticmethod
    def schedule_ts(n_steps, from_sigma=1, to_sigma=1e-2, rho=0.7):
        ramp = torch.linspace(0, 1, n_steps + 1)
        min_inv_rho = to_sigma ** (1 / rho)
        max_inv_rho = from_sigma ** (1 / rho)
        return Model.sigmas_to_ts(
            (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        )

    @staticmethod
    def sigmas_to_ts(sigmas):
        sigmas = torch.as_tensor(sigmas)
        return utils.alpha_sigma_to_t(1 - sigmas, sigmas)

    def alphas(self, ts):
        if isinstance(ts, float):
            ts = torch.tensor(ts)
        if ts.ndim == 0:
            ts = ts[None]
        if ts.ndim != 1:
            raise ValueError("t must be a scalar or a 1D tensor")
        alphas, _ = utils.t_to_alpha_sigma(ts)
        return alphas[:, None, None, None].to(self.device)

    def sigmas(self, ts):
        if isinstance(ts, float):
            ts = torch.tensor(ts)
        if ts.ndim == 0:
            ts = ts[None]
        if ts.ndim != 1:
            raise ValueError("t must be a scalar or a 1D tensor")
        _, sigmas = utils.t_to_alpha_sigma(ts)
        return sigmas[:, None, None, None].to(self.device)

    @torch.cuda.amp.autocast()
    def velocities(self, diffused, t, conditioning=None):
        x = diffusion_space.encode(diffused)
        if x.shape[1:] != self.model.shape:
            raise ValueError(
                f"Velocity diffusion model {self.name} only works well with shape {self.shape} but got {diffused.shape}"
            )
        if hasattr(self.model, "clip_model"):
            model_fn = partial(self.model, clip_embed=conditioning.squeeze(dim=1))
        else:
            model_fn = self.model

        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)

        velocities = model_fn(x, t)
        return velocities.float()

    def forward(self, diffused_images, ts, conditioning=None):
        if isinstance(ts, float) or ts.ndim == 0:
            ts = torch.full((diffused_images.shape[0],), ts).to(diffused_images)
        return Predictions(
            from_diffused_images=diffused_images,
            from_ts=ts,
            velocities=self.velocities(diffused_images, ts, conditioning),
        )

    def predictions(self, diffused_images, ts, conditioning=None):
        return self.forward(diffused_images, ts, conditioning)

    def conditioning(self, texts=None, images=None, encodings=None):
        clip_model = models.CLIP(self.model.clip_model)

        all_encodings = list()
        if texts is not None:
            all_encodings.append(clip_model.encode_texts(texts))
        if images is not None:
            all_encodings.append(clip_model.encode_images(images))
        if encodings is not None:
            all_encodings.append(encodings)
        if len(all_encodings) == 0:
            raise ValueError("Must provide at least one of texts, images, or encodings")
        return torch.stack(all_encodings, dim=0).mean(dim=0)[None]

    def diffuse(self, denoised_images, ts, noise=None):
        denoised_xs = diffusion_space.encode(denoised_images)
        if isinstance(ts, float) or ts.ndim == 0:
            ts = torch.full((denoised_xs.shape[0],), ts).to(denoised_xs)
        if noise is None:
            noise = torch.randn_like(denoised_xs)
        alphas, sigmas = self.alphas(ts), self.sigmas(ts)
        return diffusion_space.decode(denoised_xs * alphas + noise * sigmas)


VelocityDiffusion: Model = cache(Model)


def test_velocity_diffusion():
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    diffusion = models.VelocityDiffusion("yfcc_2").to(device)

    n_iterations = 3

    steps = diffusion.schedule_ts(n_iterations, from_sigma=1.0, rho=0.7)

    diffused_images = torch.randn((1, 3, 512, 512)).to(device).add(1).div(2)

    for from_ts, to_ts in zip(steps[:-1], steps[1:]):
        if (from_ts < 1.0).all():
            new_from_ts = from_ts * 1.003
            diffused_images = diffusion.predictions(
                diffused_images, from_ts
            ).noisy_reverse_step(new_from_ts)
            from_ts = new_from_ts

        predictions = diffusion.predictions(
            diffused_images,
            from_ts,
        )
        diffused_images = predictions.step(to_ts)
        diffused_images = (
            diffusion.predictions(diffused_images, to_ts)
            .correction(predictions)
            .step(to_ts)
        )


def test_conditioned_velocity_diffusion():
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    diffusion = models.VelocityDiffusion("cc12m_1_cfg").to(device)

    diffused_images = torch.randn((1, 3, 256, 256)).to(device).add(1).div(2)

    conditioning = diffusion.conditioning(texts=["hello world"])
    from_ts = 1.0

    diffused_images = diffusion.predictions(diffused_images, from_ts, conditioning)
