from typing import Optional, Literal, Union, List
from contextlib import contextmanager
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import lantern
from transformers import CLIPTokenizer, CLIPTextModel, logging
from diffusers import StableDiffusionPipeline, DDPMScheduler
import diffusers.models

import perceptor

from . import diffusion_space
from .predictions import Predictions
from .conditioning import Conditioning

try:
    from . import attention

    XFORMERS_INSTALLED = True
except ImportError:
    XFORMERS_INSTALLED = False


# @cache
class StableDiffusion(torch.nn.Module):
    def __init__(
        self,
        name: str = "runwayml/stable-diffusion-v1-5",
        fp16: bool = True,
        auth_token: Union[bool, str] = True,
        flash_attention: bool = True,
        attention_slicing: Optional[Union[int, Literal["auto"]]] = None,
    ):
        """
        Stable Diffusion text2image model.

        Args:
            name (str, optional): Name of the model. Defaults to "runwayml/stable-diffusion-v1-5". Available models are:
                - runwayml/stable-diffusion-v1-5 (512x512)
                - runwayml/stable-diffusion-inpainting (512x512)
                - CompVis/stable-diffusion-v1-4 (512x512)
                - Huggingface model id
                - Path to weights
            fp16 (bool, optional): Whether to use mixed precision. Defaults to True.
            auth_token (bool, optional): Whether to use an auth token. Defaults to True.
            flash_attention (bool, optional): Whether to use flash attention. Defaults to True.
            attention_slicing (Union[int, Literal["auto"]], optional): Number of attention steps. Defaults to None.
                Options are "auto" or an integer. Lowers VRAM usage but increases inference time.
        """
        super().__init__()
        self.name = name

        scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        if XFORMERS_INSTALLED and flash_attention:
            # monkeypatch xformers flash attention
            patch = {
                "AttentionBlock",
                "FeedForward",
                "CrossAttention",
                "SpatialTransformer",
            }

            for attribute in patch:
                setattr(
                    diffusers.models.attention, attribute, getattr(attention, attribute)
                )

        pipeline = StableDiffusionPipeline.from_pretrained(
            name,
            scheduler=scheduler,
            use_auth_token=auth_token,
            **(
                dict(
                    revision="fp16",
                    torch_dtype=torch.float16,
                )
                if fp16
                else dict()
            ),
        )

        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.feature_extractor = pipeline.feature_extractor
        self.scheduler = pipeline.scheduler

        if attention_slicing is not None:
            if attention_slicing == "auto":
                attention_slicing = 2
            slice_size = self.unet.config.attention_head_dim // attention_slicing
            self.unet.set_attention_slice(slice_size)

        self.schedule_alphas = torch.nn.Parameter(
            self.scheduler.alphas_cumprod.sqrt(), requires_grad=False
        )
        self.schedule_sigmas = torch.nn.Parameter(
            (1 - self.scheduler.alphas_cumprod).sqrt(),
            requires_grad=False,
        )

        self.vae_original_requires_grads = [
            parameter.requires_grad for parameter in self.vae.parameters()
        ]
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet.eval()
        self.unet.requires_grad_(False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def shape(self):
        return self.model.shape

    def schedule_indices(
        self, n_steps=500, from_index=999, to_index=0, rho=3.0
    ) -> lantern.Tensor:
        if from_index < to_index:
            raise ValueError("from_index must be greater than to_index")

        from_alpha, from_sigma = self.alphas(from_index), self.sigmas(from_index)
        to_alpha, to_sigma = self.alphas(to_index), self.sigmas(to_index)

        from_log_snr = torch.log(from_alpha**2 / from_sigma**2)
        to_log_snr = torch.log(to_alpha**2 / to_sigma**2)

        elucidated_from_sigma = (1 / from_log_snr.exp()).sqrt().clamp(max=150)
        elucidated_to_sigma = (1 / to_log_snr.exp()).sqrt().clamp(min=1e-3)

        ramp = torch.linspace(0, 1, n_steps + 1).to(self.device)
        min_inv_rho = elucidated_to_sigma ** (1 / rho)
        max_inv_rho = elucidated_from_sigma ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        target_log_snr = torch.log(torch.ones_like(sigmas) ** 2 / sigmas**2)

        schedule_log_snr = torch.log(
            self.schedule_alphas**2 / self.schedule_sigmas**2
        )

        assert target_log_snr.squeeze().ndim == 1
        assert schedule_log_snr.squeeze().ndim == 1
        schedule_indices = (
            (target_log_snr.squeeze()[:, None] - schedule_log_snr.squeeze()[None, :])
            .abs()
            .argmin(dim=1)
            .unique()
            .sort(descending=True)[0]
        )

        if len(schedule_indices) <= n_steps * 0.9:
            raise ValueError(
                f"Scheduled steps {len(schedule_indices)} is too far from wanted number of steps {n_steps}"
            )

        assert (schedule_indices[:-1] != schedule_indices[1:]).all()
        return torch.stack([schedule_indices[:-1], schedule_indices[1:]], dim=1)

    @torch.cuda.amp.autocast()
    def encode(
        self, images: lantern.Tensor.dims("NCHW").float(), method="mode"
    ) -> lantern.Tensor.dims("NCHW"):
        _, _, h, w = images.shape
        if h % 32 != 0:
            raise Exception(f"Height must be divisible by 32, got {h}")
        if w % 32 != 0:
            raise Exception(f"Width must be divisible by 32, got {w}")

        distribution = self.vae.encode(diffusion_space.encode(images.to(self.device)))

        if method == "sample":
            return 0.18215 * distribution.latent_dist.sample()
        elif method == "mode":
            return 0.18215 * distribution.latent_dist.mode()
        else:
            raise ValueError(f"Unknown encoding method {method}")

    @torch.cuda.amp.autocast()
    def decode(
        self, latents: lantern.Tensor.dims("NCHW").float()
    ) -> lantern.Tensor.dims("NCHW"):
        return diffusion_space.decode(self.vae.decode(latents / 0.18215).sample)

    @contextmanager
    def finetuneable_vae(self):
        """
        with diffusion_model.finetuneable_vae():
            images = diffusion_model.decode(latents)
        """
        state_dict = copy.deepcopy(self.vae.state_dict())
        try:
            for parameter, requires_grad in zip(
                self.vae.parameters(), self.vae_original_requires_grads
            ):
                parameter.requires_grad_(requires_grad)
            yield self
        finally:
            self.vae.load_state_dict(state_dict)
            self.vae.requires_grad_(False)

    @torch.cuda.amp.autocast()
    def latents(
        self, images: lantern.Tensor.dims("NCHW").float()
    ) -> lantern.Tensor.dims("NCHW"):
        return self.encode(images).float()

    @torch.cuda.amp.autocast()
    def images(
        self, latents: lantern.Tensor.dims("NCHW").float()
    ) -> lantern.Tensor.dims("NCHW"):
        return self.decode(latents).float()

    def random_diffused_latents(self, shape) -> lantern.Tensor:
        n, c, h, w = shape
        if h % 32 != 0:
            raise ValueError("Height must be divisible by 32")
        if w % 32 != 0:
            raise ValueError("Width must be divisible by 32")
        return (
            torch.randn((n, self.unet.in_channels, h // 8, w // 8)).to(self.device)
            * self.scheduler.init_noise_sigma
        )

    def indices(self, indices) -> lantern.Tensor:
        if isinstance(indices, float) or isinstance(indices, int):
            indices = torch.as_tensor(indices)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1:
            raise ValueError("indices must be a scalar or a 1-dimensional tensor")
        return indices.long().to(self.device)

    def alphas(self, indices) -> lantern.Tensor:
        return self.schedule_alphas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    def sigmas(self, indices) -> lantern.Tensor:
        return self.schedule_sigmas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    @torch.cuda.amp.autocast()
    def predicted_noise(
        self,
        diffused_latents,
        from_indices,
        conditioning: Conditioning,
    ) -> lantern.Tensor:
        predicted_noise = self.unet(
            conditioning.input(diffused_latents),
            self.indices(from_indices),
            conditioning.encodings,
        )["sample"]
        return predicted_noise.float()

    def forward(
        self,
        diffused_latents: lantern.Tensor,
        indices: lantern.Tensor,
        conditioning: Optional[Conditioning] = None,
    ) -> Predictions:
        indices = self.indices(indices)
        return Predictions(
            from_diffused_latents=diffused_latents,
            from_indices=indices,
            predicted_noise=self.predicted_noise(
                diffused_latents, indices, conditioning
            ),
            schedule_alphas=self.schedule_alphas,
            schedule_sigmas=self.schedule_sigmas,
            encode=self.encode,
            decode=self.decode,
        )

    def predictions(self, diffused_latents, indices, conditioning) -> Predictions:
        return self.forward(diffused_latents, indices, conditioning)

    def text_encodings(self, texts):
        verbosity = logging.get_verbosity()
        logging.set_verbosity_error()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        logging.set_verbosity(verbosity)

        tokenized_text = tokenizer(
            texts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = tokenized_text.input_ids

        if text_input_ids.shape[-1] > tokenizer.model_max_length:
            removed_text = tokenizer.batch_decode(
                text_input_ids[:, tokenizer.model_max_length :]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : tokenizer.model_max_length]

        return text_encoder(text_input_ids.to(self.device))[0]

    def latent_masks(self, masks):
        n, c, h, w = masks.shape
        if h % 8 != 0:
            raise ValueError("Height must be divisible by 8")
        if w % 8 != 0:
            raise ValueError("Width must be divisible by 8")
        if c != 1:
            raise ValueError("Masks must be 1-channel")
        if masks.gt(1).any() or masks.lt(0).any():
            raise ValueError("Masks must be between 0 and 1")
        return F.interpolate(
            masks.to(self.device).float(), size=(h // 8, w // 8), mode="nearest"
        )

    def conditioning(
        self,
        texts: List[str] = [""],
        inpainting_masks: Optional[lantern.Tensor.dims("NCHW")] = None,
        inpainting_images: Optional[lantern.Tensor.dims("NCHW")] = None,
        # mask_blur=4.0,  # TODO
    ) -> Conditioning:
        """
        Create a conditioning object from a list of texts. Unconditional is an empty string.

        Args:
            texts: A list of texts to condition on. Unconditional is an empty string
            inpainting_masks: A tensor of masks to condition on. Must be 1-channel and between 0 and 1
            inpainting_images: A tensor of images to condition on. Must be 3-channel and between 0 and 1
        """
        if self.name == "runwayml/stable-diffusion-inpainting":
            inpainting_latent_masks = self.latent_masks(inpainting_masks)
            inpainting_latents = self.latents(
                inpainting_images * inpainting_masks.le(0.5)
                + 0.5 * inpainting_masks.gt(0.5).float()
            )
        else:
            inpainting_latent_masks = None
            inpainting_latents = None

        return Conditioning(
            model_name=self.name,
            encodings=self.text_encodings(texts),
            inpainting_latent_masks=inpainting_latent_masks,
            inpainting_latents=inpainting_latents,
        )

    def diffuse_latents(self, denoised_latents, indices, noise=None) -> lantern.Tensor:
        indices = self.indices(indices)
        if noise is None:
            noise = torch.randn_like(denoised_latents)
        alphas, sigmas = self.alphas(indices), self.sigmas(indices)
        return denoised_latents * alphas + noise * sigmas

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def sample(
        self,
        text,
        from_index=999,
        to_index=0,
        n_steps=50,
        guidance_scale=7.0,
        n_resample=0,
        init_image=None,
        inpainting_mask=None,
        replace_diffused=False,
    ):
        neutral_conditioning = self.conditioning(
            texts=[""], inpainting_masks=inpainting_mask, inpainting_images=init_image
        )
        positive_conditioning = self.conditioning(
            texts=[text], inpainting_masks=inpainting_mask, inpainting_images=init_image
        )

        schedule_indices = self.schedule_indices(
            from_index=from_index, to_index=to_index, n_steps=n_steps
        )
        from_index = schedule_indices[0, 0]

        if init_image is None:
            if from_index != 999:
                raise ValueError("init_image must be provided if from_index < 999")
            diffused_latents = self.random_diffused_latents((1, 3, 512, 512))
        else:
            init_latents = self.latents(init_image)
            diffused_latents = self.diffuse_latents(init_latents, from_index)

        for from_index, to_index in tqdm(schedule_indices):
            for _ in range(n_resample):
                unconditioned_predictions = self.predictions(
                    diffused_latents,
                    from_index,
                    neutral_conditioning,
                )
                positive_predictions = self.predictions(
                    diffused_latents,
                    from_index,
                    positive_conditioning,
                )
                strongly_positive_predictions = (
                    unconditioned_predictions.classifier_free_guidance(
                        positive_predictions, guidance_scale=guidance_scale
                    )
                )
                diffused_latents = strongly_positive_predictions.resample(to_index)

            unconditioned_predictions = self.predictions(
                diffused_latents,
                from_index,
                neutral_conditioning,
            )
            positive_predictions = self.predictions(
                diffused_latents,
                from_index,
                positive_conditioning,
            )
            strongly_positive_predictions = (
                unconditioned_predictions.classifier_free_guidance(
                    positive_predictions, guidance_scale=guidance_scale
                )
            )
            diffused_latents = strongly_positive_predictions.step(to_index)

            if replace_diffused and inpainting_mask is not None:
                # this is peeking into the original masked image
                diffused_latents = self.diffuse_latents(init_latents, to_index) * (
                    1 - positive_conditioning.inpainting_latent_masks
                ) + diffused_latents * (positive_conditioning.inpainting_latent_masks)

            yield positive_predictions

        yield self.predictions(
            diffused_latents,
            to_index,
            positive_conditioning,
        )


def test_stable_diffusion_attention_slicing():
    diffusion_model = StableDiffusion(attention_slicing="auto").cuda()
    diffused_latents = diffusion_model.random_diffused_latents((1, 3, 256, 256))
    diffusion_model.predictions(diffused_latents, 100, diffusion_model.conditioning())


def test_stable_diffusion():
    for predictions in (
        StableDiffusion().cuda().sample("photograph of a playful cat", to_index=20)
    ):
        pass
    perceptor.utils.pil_image(predictions.denoised_images.clamp(0, 1)).save(
        "tests/stable_diffusion.png"
    )


def test_stable_diffusion_init_image():
    import requests
    from PIL import Image
    import torchvision.transforms.functional as TF

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    init_image = TF.to_tensor(
        Image.open(requests.get(image_url, stream=True).raw).resize((512, 512))
    )[None].cuda()

    for predictions in (
        StableDiffusion()
        .cuda()
        .sample(
            "photograph of playful lions",
            from_index=500,
            to_index=20,
            init_image=init_image,
            n_resample=4,
            guidance_scale=3,
        )
    ):
        pass
    perceptor.utils.pil_image(predictions.denoised_images.clamp(0, 1)).save(
        "tests/stable_diffusion_init_image.png"
    )


def test_stable_diffusion_inpainting():
    import requests
    from PIL import Image
    import torch
    import torchvision.transforms.functional as TF
    from perceptor import utils

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    init_image = TF.to_tensor(
        Image.open(requests.get(image_url, stream=True).raw).resize((512, 512))
    )[None].cuda()
    inpainting_mask = torch.zeros_like(init_image[:, :1])
    inpainting_mask[:, :, :, 128:] = 1.0

    torch.set_grad_enabled(False)
    diffusion_model = StableDiffusion(
        "runwayml/stable-diffusion-inpainting", fp16=False
    ).cuda()

    for predictions in diffusion_model.sample(
        "photograph of playful lions",
        from_index=600,
        to_index=20,
        n_steps=50,
        init_image=init_image,
        inpainting_mask=inpainting_mask,
        replace_diffused=True,
        n_resample=4,
    ):
        pass
    utils.pil_image(predictions.denoised_images.clamp(0, 1)).save(
        "tests/stable_diffusion_inpainting.png"
    )


def test_stable_diffusion_step():
    from diffusers import DDIMScheduler

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    for model_name in [
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
    ]:
        batch_size = 1
        height = 512
        width = 512

        from_index = 999
        to_index = 998

        texts = ["painting of a dog"]

        diffusion_model = StableDiffusion(model_name).to(device)

        diffused_latents = diffusion_model.random_diffused_latents((1, 3, 512, 512))

        conditioning = diffusion_model.conditioning(texts)
        predictions = diffusion_model.predictions(
            diffused_latents, from_index, conditioning
        )
        compare_next_diffused_latents = predictions.step(to_index)

        del diffusion_model

        # compare with diffusers
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        scheduler.set_timesteps(1000)
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_auth_token=True,
        ).to(device)

        latents_shape = (batch_size, pipeline.unet.in_channels, height // 8, width // 8)
        assert latents_shape == diffused_latents.shape

        tokenized_text = pipeline.tokenizer(
            texts,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipeline.text_encoder(
            tokenized_text.input_ids.to(pipeline.device)
        )[0]

        assert text_embeddings.shape == conditioning.encodings.shape
        assert torch.allclose(
            text_embeddings[0, 0], conditioning.encodings[0, 0], atol=1e-3
        )
        assert torch.allclose(text_embeddings, conditioning.encodings, atol=1e-3)

        index = next(iter(pipeline.scheduler.timesteps))
        assert from_index == index

        predicted_noise = pipeline.unet(
            diffused_latents, index, encoder_hidden_states=text_embeddings
        )["sample"]

        assert torch.allclose(predictions.predicted_noise, predicted_noise, atol=5e-3)

        next_diffused_latents = pipeline.scheduler.step(
            predicted_noise, index, diffused_latents
        )["prev_sample"]

        assert torch.allclose(
            next_diffused_latents[0, 0, 0, 0],
            compare_next_diffused_latents[0, 0, 0, 0],
            atol=1e-3,
        )
        assert (
            next_diffused_latents.sub(compare_next_diffused_latents).abs().max() <= 1e-3
        )
