from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch
from basicsr.utils.download_util import load_file_from_url

from .ldm.util import instantiate_from_config
from perceptor.utils import cache
from . import diffusion_space

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@cache
class Text2Image(torch.nn.Module):
    def __init__(self, guidance_scale=5, eta=0.0):
        super().__init__()
        self.eta = eta
        self.guidance_scale = guidance_scale

        url_ckpt = "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt"

        config_path = CONFIG_DIR / "latent-diffusion" / "txt2img-1p4B-eval.yaml"
        checkpoint_path = load_file_from_url(
            url_ckpt, "models", file_name="latent-diffusion-txt2img-f8-large.ckpt"
        )

        config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False
        )
        self.model.requires_grad_(False)
        self.model.cuda()
        self.model.eval()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @staticmethod
    def latent_shape(height, width):
        return [4, height // 8, width // 8]

    def forward(self, latents, conditioning, index):
        return self.denoise(latents, conditioning, index)

    def velocity(self, latents, conditioning, index):
        raise NotImplementedError()

    def random_latents(self, images_shape):
        return torch.randn(
            images_shape[0], *self.latent_shape(*images_shape[-2:]), device=self.device
        )

    def latents(self, images):
        encoder_posterior = self.model.encode_first_stage(
            diffusion_space.encode(images)
        )
        return self.model.get_first_stage_encoding(encoder_posterior)

    def conditioning(self, text_prompts, negative_text_prompts=[""]):
        """Encode images (0-1) to latent space"""
        return torch.cat(
            [
                self.model.get_learned_conditioning(text_prompts),
                self.model.get_learned_conditioning(negative_text_prompts),
            ],
            dim=0,
        )

    def diffuse(self, latents, index, noise=None):
        """Unclear what the first argument should be. Conditioning works and latents also works okay"""
        if noise is None:
            noise = torch.randn_like(latents)

        return self.model.q_sample(x_start=latents, t=self.ts(index), noise=noise)

    def denoise(self, latents, index, conditioning=None, eps=None):
        """Predict denoised latents"""
        if eps is None:
            eps = self.eps(latents, index, conditioning)

        return (
            latents - self.sqrt_one_minus_alphas_cumprod(index) * eps
        ) / self.alphas_cumprod(index).sqrt()

    def images(self, latents):
        """Decode from latent space to image"""
        return diffusion_space.decode(self.model.decode_first_stage(latents))

    def ts(self, index):
        return torch.tensor([index], device=self.device, dtype=torch.long)

    def alphas_cumprod(self, index):
        return self.model.alphas_cumprod[index].to(self.device)[None, None, None, None]

    def sqrt_one_minus_alphas_cumprod(self, index):
        return self.model.sqrt_one_minus_alphas_cumprod[index].to(self.device)[
            None, None, None, None
        ]

    def step(
        self, from_latents, predicted_denoised_latents, from_index, to_index, noise=None
    ):
        if to_index > from_index:
            raise ValueError("to_index must be smaller than from_index")
        if noise is None:
            noise = torch.randn_like(predicted_denoised_latents)

        # if quantize_denoised:
        #     predicted_denoised_latents, _, *_ = self.model.first_stage_model.quantize(predicted_denoised_latents)

        from_alphas_cumprod = self.alphas_cumprod(from_index)
        to_alphas_cumprod = self.alphas_cumprod(to_index)
        from_sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod(
            from_index
        )

        to_sigmas = self.eta * torch.sqrt(
            (1 - to_alphas_cumprod)
            / (1 - from_alphas_cumprod)
            * (1 - from_alphas_cumprod / to_alphas_cumprod)
        )

        eps = (
            from_latents - predicted_denoised_latents * from_alphas_cumprod.sqrt()
        ) / from_sqrt_one_minus_alphas_cumprod

        dir_xt = (1.0 - to_alphas_cumprod - to_sigmas**2).sqrt() * eps

        to_z = (
            to_alphas_cumprod.sqrt() * predicted_denoised_latents
            + dir_xt
            + to_sigmas * noise
        )
        return to_z

    def eps(self, latents, index, conditioning):
        if index >= 1000:
            raise ValueError("index must be less than 1000")

        if self.guidance_scale is None or self.guidance_scale == 1.0:
            eps = self.model.apply_model(latents, self.ts(index), conditioning)
        else:
            # save vRAM by splitting this in two?
            eps_conditioned, eps_negative = self.model.apply_model(
                torch.cat([latents] * 2),
                torch.cat([self.ts(index)] * 2),
                conditioning,
            ).chunk(2)

            eps = eps_negative + self.guidance_scale * (eps_conditioned - eps_negative)
        return eps
