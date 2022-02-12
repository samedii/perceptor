from functools import partial
from multiprocessing.sharedctypes import Value
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
        add_sigma = (
            to_sigma.square() - from_sigma.square()
        ).sqrt()  # does alpha affect this?
        return (
            (
                diffused_images.mul(2).sub(1) * to_alpha / from_alpha
                + torch.randn_like(diffused_images) * add_sigma
            )
            .add(1)
            .div(2)
        )

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

        def guide_wrapper(x, t, pred, clip_embed=None):
            nonlocal predicted_images
            predicted_images = (pred + 1) / 2
            guide(predicted_images)
            cond_grad = -x.grad
            return cond_grad * 500

        if hasattr(self.model, "clip_model"):
            extra_args = dict(clip_embed=self.encodings)
        else:
            extra_args = dict()

        if guide is None:
            model_fn = self.model
        else:
            model_fn = sampling.make_cond_model_fn(self.model, guide_wrapper)

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

    def predict_denoised(self, diffused_images, from_noise):
        """Predict the denoised images."""
        pass

    def single_step(self, diffused_images, from_noise, to_noise):
        """Perform a single step of the vanilla diffusion process."""
        pass
