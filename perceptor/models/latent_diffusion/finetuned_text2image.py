# finetuned model from https://github.com/Jack000/glid-3-xl
import sys
from pathlib import Path
import pickle
from omegaconf import OmegaConf
import torch
import torch
from basicsr.utils.download_util import load_file_from_url

from . import ldm
from perceptor.models.clip import CLIP
from .finetuned.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from .finetuned.encoders.modules import BERTEmbedder
from .ldm.util import instantiate_from_config
from perceptor.utils import cache
from . import diffusion_space

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "ldm":
            return ldm
        return super().find_class(module, name)


@cache
class FinetunedText2Image(torch.nn.Module):
    def __init__(self, guidance_scale=5, eta=0.0):
        super().__init__()
        self.eta = eta
        self.clip_model_name = "ViT-L/14"
        self.guidance_scale = guidance_scale

        autoencoder_url_ckpt = "https://dall-3.com/models/glid-3-xl/kl-f8.pt"
        autoencoder_path = load_file_from_url(
            autoencoder_url_ckpt, "models", file_name="latent-diffusion-kl-f8.pt"
        )

        bert_url_ckpt = "https://dall-3.com/models/glid-3-xl/bert.pt"
        bert_path = load_file_from_url(
            bert_url_ckpt, "models", file_name="latent-diffusion-bert.pt"
        )

        url_finetune_ckpt = "https://dall-3.com/models/glid-3-xl/finetune.pt"
        finetune_path = load_file_from_url(
            url_finetune_ckpt,
            "models",
            file_name="latent-diffusion-txt2img-finetune.pt",
        )

        model_state_dict = torch.load(finetune_path, map_location="cpu")
        self.model_params = {
            "attention_resolutions": "32,16,8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",
            "image_size": 32,
            "learn_sigma": False,
            "noise_schedule": "linear",
            "num_channels": 320,
            "num_heads": 8,
            "num_res_blocks": 2,
            "resblock_updown": False,
            "use_fp16": False,
            "use_scale_shift_norm": False,
            "clip_embed_dim": 768 if "clip_proj.weight" in model_state_dict else None,
            "image_condition": True
            if model_state_dict["input_blocks.0.0.weight"].shape[1] == 8
            else False,
            "super_res_condition": True
            if "external_block.0.0.weight" in model_state_dict
            else False,
        }

        assert self.model_params["clip_embed_dim"]

        model_config = model_and_diffusion_defaults()
        model_config.update(self.model_params)

        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.eval().requires_grad_(False)

        if model_config["use_fp16"]:
            self.model.convert_to_fp16()
        else:
            self.model.convert_to_fp32()

        # hack for pickle
        sys.modules["ldm"] = ldm
        self.autoencoder = torch.load(autoencoder_path, map_location="cpu")
        # self.autoencoder = Unpickler(Path(autoencoder_path).open("rb")).load()
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        self.bert = BERTEmbedder(1280, 32)
        sd = torch.load(bert_path, map_location="cpu")
        self.bert.load_state_dict(sd)
        self.bert.half().eval().requires_grad_(False)

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
        return (
            self.autoencoder.encode(diffusion_space.encode(images)).sample() * 0.18215
        )

    def conditioning(
        self,
        text_prompts,
        negative_text_prompts=[""],
        images=None,
        negative_images=None,
    ):
        """Encode images (0-1) to latent space"""
        text_embedding = self.bert.encode(text_prompts).to(self.device).float()
        text_embedding_blank = (
            self.bert.encode(negative_text_prompts).to(self.device).float()
        )
        clip_model = CLIP(self.clip_model_name)
        # embeddings should not be normalized
        embedding_clip = clip_model.model.encode_texts(text_prompts).squeeze(1)
        embedding_clip_negative = clip_model.model.encode_texts(
            negative_text_prompts
        ).squeeze(1)
        if images is not None:
            embedding_clip = torch.cat(
                [
                    embedding_clip,
                    clip_model.model.encode_image(images),
                ],
                dim=0,
            )
        if negative_images is not None:
            embedding_clip_negative = torch.cat(
                [
                    embedding_clip_negative,
                    clip_model.model.encode_image(negative_images),
                ],
                dim=0,
            )
        return dict(
            context=torch.cat([text_embedding, text_embedding_blank], dim=0).float(),
            clip_embed=torch.cat([embedding_clip, embedding_clip_negative], dim=0)
            .float()
            .mean(dim=0)[None],
        )

    def diffuse(self, latents, index, noise=None):
        """Unclear what the first argument should be. Conditioning works and latents also works okay"""
        if noise is None:
            noise = torch.randn_like(latents)

        return self.diffusion.q_sample(x_start=latents, t=self.ts(index), noise=noise)

    def denoise(self, latents, index, conditioning=None, eps=None):
        """Predict denoised latents"""
        if eps is None:
            eps = self.eps(latents, index, conditioning)

        return (
            latents - self.sqrt_one_minus_alphas_cumprod(index) * eps
        ) / self.alphas_cumprod(index).sqrt()

    def images(self, latents):
        """Decode from latent space to image"""
        return diffusion_space.decode(self.autoencoder.decode(latents / 0.18215))

    def ts(self, index):
        return torch.tensor([index], device=self.device, dtype=torch.long)

    def alphas_cumprod(self, index):
        return (
            torch.as_tensor(self.diffusion.alphas_cumprod[index])
            .to(self.device)[None, None, None, None]
            .float()
        )

    def sqrt_one_minus_alphas_cumprod(self, index):
        return (
            torch.as_tensor(self.diffusion.sqrt_one_minus_alphas_cumprod[index])
            .to(self.device)[None, None, None, None]
            .float()
        )

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

        eps_conditioned, eps_unconditioned = self.model(
            torch.cat([latents] * 2),
            torch.cat([self.ts(index)] * 2),
            conditioning["context"],
            conditioning["clip_embed"],
        ).chunk(2)

        return eps_unconditioned + self.guidance_scale * (
            eps_conditioned - eps_unconditioned
        )
