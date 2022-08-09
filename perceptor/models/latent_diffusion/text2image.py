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

    def schedule_indices(self, from_index=999, to_index=50, n_steps=None):
        if from_index < to_index:
            raise ValueError("from_index must be greater than to_index")
        if n_steps is None:
            n_steps = (from_index - to_index) // 2
        schedule_indices = torch.linspace(from_index, to_index, n_steps).long()
        from_indices = schedule_indices[:-1]
        to_indices = schedule_indices[1:]
        if (from_indices == to_indices).any():
            raise ValueError("Schedule indices must be unique")
        return torch.stack([from_indices, to_indices], dim=1)

    @staticmethod
    def latent_shape(height, width):
        return [4, height // 8, width // 8]

    def forward(self, latents, conditioning, index):
        return self.denoise(latents, conditioning, index)

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
        return torch.cat(
            [
                self.model.get_learned_conditioning(text_prompts),
                self.model.get_learned_conditioning(negative_text_prompts),
            ],
            dim=0,
        )

    def diffuse(self, latents, index, noise=None):
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


def test_text2image():
    from tqdm import tqdm
    import perceptor

    model = perceptor.models.latent_diffusion.Text2Image().cuda()
    conditioning = model.conditioning(["photograph of a playful cat"])
    diffused_latents = model.random_latents((1, 3, 512, 512)).cuda()

    for from_index, to_index in tqdm(model.schedule_indices(to_index=50, n_steps=50)):
        denoised_latents = model.denoise(diffused_latents, from_index, conditioning)
        diffused_latents = model.step(
            diffused_latents, denoised_latents, from_index, to_index
        )
    denoised_latents = model.denoise(diffused_latents, to_index, conditioning)
    images = model.images(denoised_latents)
    perceptor.utils.pil_image(images).save("tests/text2image.png")


def test_text2image_latents():
    model = Text2Image().cpu()
    assert (
        model.random_latents((1, 3, 512, 512)).shape
        == model.latents(torch.randn((1, 3, 512, 512))).shape
    )
