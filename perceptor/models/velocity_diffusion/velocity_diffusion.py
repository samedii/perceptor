from functools import partial
import torch
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from perceptor import models
from .models import get_model
from .model_urls import model_urls
from . import diffusion_space, utils


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
    def alphas(t):
        if t.ndim == 0:
            t = t[None]
        alphas, _ = utils.t_to_alpha_sigma(t)
        return alphas[:, None, None, None]

    @staticmethod
    def sigmas(t):
        if t.ndim == 0:
            t = t[None]
        _, sigmas = utils.t_to_alpha_sigma(t)
        return sigmas[:, None, None, None]

    def forward(self, images, t, conditioning=None):
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((images.shape[0],), t).to(images)
        return self.denoise(images, t, conditioning=conditioning)

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

    @torch.cuda.amp.autocast()
    def velocity(self, diffused, t, conditioning=None):
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

        velocity = model_fn(x, t)
        return velocity.float()

    @staticmethod
    def x(denoised, noise, t):
        pred = diffusion_space.encode(denoised)
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((pred.shape[0],), t).to(pred)
        alphas, sigmas = Model.alphas(t), Model.sigmas(t)
        return diffusion_space.decode(pred * alphas + noise * sigmas)

    def denoise(
        self, diffused, t, conditioning=None, eps=None, threshold_quantile=0.95
    ):
        x = diffusion_space.encode(diffused)
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)

        alphas, sigmas = self.alphas(t), self.sigmas(t)
        if eps is None:
            eps = self.eps(diffused, t, conditioning)

        pred = x * alphas - (eps - x * sigmas) * sigmas / alphas

        if threshold_quantile is not None:
            dynamic_threshold = torch.quantile(
                pred.flatten(start_dim=1).mul(2).sub(1).abs(), threshold_quantile, dim=1
            ).clamp(min=1.0)
            pred = (
                pred
                - pred.detach()
                + pred.detach().clamp(-dynamic_threshold, dynamic_threshold)
            )

        return diffusion_space.decode(pred)

    @staticmethod
    def diffuse(images, t, noise=None):
        x0 = diffusion_space.encode(images)
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x0.shape[0],), t).to(x0)
        if noise is None:
            noise = torch.randn_like(x0)
        alphas, sigmas = Model.alphas(t), Model.sigmas(t)
        return diffusion_space.decode(x0 * alphas + noise * sigmas)

    def eps(self, diffused, t, conditioning=None):
        x = diffusion_space.encode(diffused)
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)
        velocity = self.velocity(diffused, t, conditioning)
        return x * self.sigmas(t) + velocity * self.alphas(t)

    def forced_eps(self, from_diffused, t, denoised):
        from_x = diffusion_space.encode(from_diffused)
        pred = diffusion_space.encode(denoised)

        alphas, sigmas = self.alphas(t), self.sigmas(t)
        return (from_x * alphas**2 - pred * alphas) / sigmas + from_x * sigmas

    def step(self, from_diffused, from_t, to_t, denoised=None, noise=None, eta=0.0):
        if denoised is None:
            denoised = self.denoise(from_diffused, from_t)

        from_x = diffusion_space.encode(from_diffused)
        pred = diffusion_space.encode(denoised)
        if noise is None:
            noise = torch.randn_like(from_x)

        from_alphas, from_sigmas = self.alphas(from_t), self.sigmas(from_t)
        to_alphas, to_sigmas = self.alphas(to_t), self.sigmas(to_t)

        velocity = (from_x * from_alphas - pred) / from_sigmas
        eps = from_x * from_sigmas + velocity * from_alphas

        if eta > 0.0:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = (
                eta
                * (to_sigmas**2 / from_sigmas**2).sqrt()
                * (1 - from_alphas**2 / to_alphas**2).sqrt()
            )
            adjusted_sigma = (to_sigmas**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            to_x = pred * to_alphas + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            to_x += noise * ddim_sigma
        else:
            to_x = pred * to_alphas + eps * to_sigmas

        return diffusion_space.decode(to_x)

    def resample(
        self, from_diffused, from_t, to_t, conditioning=None, eps=None, noise=None
    ):
        diffused_xs = diffusion_space.encode(from_diffused)

        if eps is None:
            eps = self.eps(from_diffused, from_t, conditioning)

        denoised = self.denoise(from_diffused, from_t, conditioning, eps)
        eps = self.forced_eps(from_diffused, from_t, denoised)

        if noise is None:
            noise = torch.randn_like(diffused_xs)

        replaced_noise_sigma = (
            self.sigmas(from_t) ** 2 - self.sigmas(to_t) ** 2
        ).sqrt()
        diffused_xs = (
            diffused_xs
            + eps * (self.sigmas(to_t) - self.sigmas(from_t))
            + torch.randn_like(eps) * replaced_noise_sigma
        )
        return diffusion_space.decode(diffused_xs)

    def corrected_step(
        self,
        from_diffused,
        from_t,
        to_t,
        conditioning=None,
        from_denoised=None,
        to_denoised=None,
    ):
        if from_denoised is None:
            from_denoised = self.denoise(from_diffused, from_t, conditioning)

        to_diffused = self.step(from_diffused, from_t, to_t, from_denoised)

        if to_denoised is None:
            to_denoised = self.denoise(to_diffused, to_t, conditioning)

        from_eps = self.forced_eps(from_diffused, from_t, from_denoised)
        to_eps = self.forced_eps(to_diffused, to_t, to_denoised)

        eps = (to_eps + from_eps) / 2
        corrected_denoised = self.denoise(from_diffused, from_t, conditioning, eps)
        return self.step(from_diffused, from_t, to_t, corrected_denoised)


VelocityDiffusion = cache(Model)
