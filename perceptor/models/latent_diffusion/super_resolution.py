from omegaconf import OmegaConf
import torch
import einops
import torch
from basicsr.utils.download_util import load_file_from_url

from perceptor.transforms.resize import resize
from .ldm.util import instantiate_from_config
from perceptor.utils import cache
from . import diffusion_space


# @cache
class SuperResolution(torch.nn.Module):
    def __init__(self, eta=1.0, convolutional=False, kernel_size=128, stride=64):
        super().__init__()
        self.eta = eta

        url_conf = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        url_ckpt = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"

        config_path = load_file_from_url(
            url_conf, "models", file_name="sharpen-colab.yaml"
        )
        checkpoint_path = load_file_from_url(
            url_ckpt, "models", file_name="sharpen-colab.ckpt"
        )

        config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False
        )
        self.model.requires_grad_(False)
        self.model.cuda()
        self.model.eval()

        self.up_f = 4
        if convolutional:
            vqf = 4
            self.model.split_input_params = {
                "ks": (kernel_size, kernel_size),
                "stride": (stride, stride),
                "vqf": vqf,
                "patch_distributed_vq": True,
                "tie_braker": False,
                "clip_max_weight": 0.5,
                "clip_min_weight": 0.01,
                "clip_max_tie_weight": 0.5,
                "clip_min_tie_weight": 0.01,
            }
        else:
            if hasattr(self.model, "split_input_params"):
                delattr(self.model, "split_input_params")

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, latents, conditioning, index):
        return self.denoise(latents, conditioning, index)

    def velocity(self, latents, conditioning, index):
        raise NotImplementedError()

    def upsample(self, images):
        return resize(images, out_shape=[s * self.up_f for s in images.shape[-2:]])

    def latents(self, images):
        """Encode images (0-1) to latent space"""
        example = dict(
            LR_image=diffusion_space.encode(
                einops.rearrange(
                    resize(
                        images, out_shape=[s // self.up_f for s in images.shape[-2:]]
                    ),
                    "1 c h w -> 1 h w c",
                )
            ),
            image=einops.rearrange(images, "1 c h w -> 1 h w c"),
        )

        latents, conditioning = self.model.get_input(
            example,
            self.model.first_stage_key,
            return_first_stage_outputs=False,
            force_c_encode=not (
                hasattr(self.model, "split_input_params")
                and self.model.cond_stage_key == "coordinates_bbox"
            ),
            return_original_cond=False,
        )
        return latents

    def conditioning(self, images):
        latents, conditioning = self.latents(images)
        return conditioning

    def diffuse(self, latents, index, noise=None):
        if noise is None:
            noise = torch.randn_like(latents)

        return self.model.q_sample(x_start=latents, t=self.ts(index), noise=noise)

    def denoise(self, latents, conditioning, index, eps=None):
        """Predict denoised latents"""
        if index >= 1000:
            raise ValueError("index must be less than 1000")

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

        return self.model.apply_model(latents, self.ts(index), conditioning)
