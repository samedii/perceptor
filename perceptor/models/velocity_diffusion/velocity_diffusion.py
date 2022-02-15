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

    def diffuse(self, diffused_images, from_noise=0, to_noise=1):
        """Adds noise to images."""
        from_alpha, from_sigma = utils.t_to_alpha_sigma(torch.tensor(from_noise))
        to_alpha, to_sigma = utils.t_to_alpha_sigma(torch.tensor(to_noise))
        # add_sigma = (
        #     to_sigma.square() - from_sigma.square()
        # ).sqrt()  # does alpha affect this?
        # return (
        #     (
        #         diffused_images.mul(2).sub(1) * to_alpha / from_alpha
        #         + torch.randn_like(diffused_images) * add_sigma
        #     )
        #     .add(1)
        #     .div(2)
        # )
        return (
            (
                diffused_images.mul(2).sub(1) * to_alpha / from_alpha
                + torch.randn_like(diffused_images) * (1 - to_alpha / from_alpha)
            )
            .add(1)
            .div(2)
        )
        # linear interpolate to wanted randn instead?

    def inverse(
        self,
        diffused_images,
        from_noise,
        to_noise,
        n_steps,
        guide=None,
        method="plms",
        eta=None,
    ):
        """Inverse the diffusion process."""

        predicted_images = None

        def guided_model_fn(x, t, clip_embed=None):
            nonlocal predicted_images

            alphas, sigmas = utils.t_to_alpha_sigma(t)
            diffused_images = (x + 1) / 2

            with torch.enable_grad():
                diffused_images.requires_grad_(True)
                grad_x = diffused_images * 2 - 1
                velocity = self.model(grad_x, t, **extra_args)
                pred = (
                    x * alphas[:, None, None, None]
                    - velocity * sigmas[:, None, None, None]
                )
                predicted_images = (pred + 1) / 2

                @torch.no_grad()
                def guided_velocity_fn(diffused_images_grad):
                    guided_velocity = (
                        velocity.detach()
                        + diffused_images_grad
                        * (sigmas[:, None, None, None] / alphas[:, None, None, None])
                        * 500
                    )
                    guided_pred = (
                        x * alphas[:, None, None, None]
                        - guided_velocity * sigmas[:, None, None, None]
                    )
                    guided_images = (guided_pred + 1) / 2
                    return guided_velocity, guided_images

                @torch.no_grad()
                def forced_velocity_fn(replaced_predicted):
                    replaced_pred = replaced_predicted * 2 - 1
                    x = diffused_images * 2 - 1
                    forced_velocity = (
                        x * alphas[:, None, None, None] - replaced_pred
                    ) / sigmas[:, None, None, None]
                    return forced_velocity

                guided_diffused_images, guided_velocity = guide(
                    diffused_images,
                    predicted_images,
                    alphas,
                    guided_velocity_fn,
                    forced_velocity_fn,
                )
            diffused_images.requires_grad_(False)
            guided_x = guided_diffused_images * 2 - 1
            x.copy_(guided_x)  # hack to update x in calling function
            return guided_velocity

        if hasattr(self.model, "clip_model"):
            extra_args = dict(clip_embed=self.encodings)
        else:
            extra_args = dict()

        if guide is None:
            model_fn = self.model
        else:
            model_fn = guided_model_fn

        if method == "ddpm":
            sampling_fn = partial(sampling.sample, eta=1)
        elif method == "ddim":
            if eta is None:
                eta = 0
            sampling_fn = partial(sampling.sample, eta=eta)
        elif method == "prk":
            sampling_fn = sampling.prk_sample
        elif method == "plms":
            sampling_fn = sampling.plms_sample
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        t = torch.linspace(from_noise, to_noise, n_steps + 1)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t).to(diffused_images)
        for x, eps, pred in sampling_fn(
            model_fn,
            diffused_images * 2 - 1,
            steps,
            extra_args=extra_args,
        ):
            yield (pred + 1) / 2 if guide is None else predicted_images

    def predict_denoised(self, diffused_images, noise=0):
        """Predict the denoised images."""
        noise = torch.full(diffused_images.shape[:1], noise).to(diffused_images)

        if hasattr(self.model, "clip_model"):
            model_fn = partial(self.model, clip_embed=self.encodings)
        else:
            model_fn = self.model

        x = diffused_images * 2 - 1
        velocity = model_fn(x, noise)
        alphas, sigmas = utils.t_to_alpha_sigma(noise)
        pred = x * alphas[:, None, None, None] - velocity * sigmas[:, None, None, None]
        return (pred + 1) / 2

    def single_step(self, diffused_images, images, from_noise, to_noise):
        """Perform a single step of the vanilla diffusion process."""
        if hasattr(self.model, "clip_model"):
            model_fn = partial(self.model, clip_embed=self.encodings)
        else:
            model_fn = self.model

        from_alphas, from_sigmas = utils.t_to_alpha_sigma(torch.tensor(from_noise))
        to_alphas, to_sigmas = utils.t_to_alpha_sigma(to_noise)

        from_x = diffused_images * 2 - 1
        v = model_fn(
            from_x,
            torch.full(diffused_images.shape[:1], from_noise).to(diffused_images),
        )
        pred = from_x * from_alphas - v * from_sigmas
        eps = from_x * from_sigmas + v * from_alphas

        to_x = pred * to_alphas + eps * to_sigmas
        return (to_x + 1) / 2
