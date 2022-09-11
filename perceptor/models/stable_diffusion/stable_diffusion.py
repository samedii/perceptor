from typing import Optional
from contextlib import contextmanager
import copy
import torch
import lantern
from transformers import CLIPTokenizer, CLIPTextModel, logging
from diffusers import StableDiffusionPipeline, DDPMScheduler

from perceptor.utils import cache
from . import diffusion_space
from .predictions import Predictions
from .conditioning import Conditioning


class Model(torch.nn.Module):
    def __init__(
        self, name="CompVis/stable-diffusion-v1-4", fp16=False, auth_token=True
    ):
        """
        Args:
            name: The name of the model. Available models are:
                - CompVis/stable-diffusion-v1-4 (512x512)
                - CompVis/stable-diffusion-v1-3 (512x512)
                - CompVis/stable-diffusion-v1-2 (512x512)
                - CompVis/stable-diffusion-v1-1 (256x256)
        """
        super().__init__()
        self.name = name

        scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        pipeline = StableDiffusionPipeline.from_pretrained(
            name,
            scheduler=scheduler,
            use_auth_token=auth_token,
            **dict(
                revision="fp16",
                torch_dtype=torch.float16,
            )
            if fp16
            else dict(),
        )

        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.feature_extractor = pipeline.feature_extractor
        self.scheduler = pipeline.scheduler

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

    def schedule_indices(self, n_steps=500, from_index=999, to_index=0, rho=3.0):
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

        assert len(schedule_indices) >= n_steps * 0.9

        assert (schedule_indices[:-1] != schedule_indices[1:]).all()
        return torch.stack([schedule_indices[:-1], schedule_indices[1:]], dim=1)

    def encode(
        self, images: lantern.Tensor.dims("NCHW").float()
    ) -> lantern.Tensor.dims("NCHW"):
        _, _, h, w = images.shape
        if h % 32 != 0:
            raise Exception(f"Height must be divisible by 32, got {h}")
        if w % 32 != 0:
            raise Exception(f"Width must be divisible by 32, got {w}")
        return (
            0.18215
            * self.vae.encode(diffusion_space.encode(images.to(self.device))).sample()
        )

    def decode(
        self, latents: lantern.Tensor.dims("NCHW").float()
    ) -> lantern.Tensor.dims("NCHW"):
        return diffusion_space.decode(self.vae.decode(latents / 0.18215))

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

    def random_diffused_latents(self, shape):
        n, c, h, w = shape
        if h % 8 != 0:
            raise ValueError("Height must be divisible by 32")
        if w % 8 != 0:
            raise ValueError("Width must be divisible by 32")
        return torch.randn((n, self.unet.in_channels, h // 8, w // 8)).to(self.device)

    def indices(self, indices):
        if isinstance(indices, float) or isinstance(indices, int):
            indices = torch.as_tensor(indices)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1:
            raise ValueError("indices must be a scalar or a 1-dimensional tensor")
        return indices.long().to(self.device)

    def alphas(self, indices):
        return self.schedule_alphas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    def sigmas(self, indices):
        return self.schedule_sigmas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    @torch.cuda.amp.autocast()
    def predicted_noise(
        self,
        diffused_latents,
        from_indices,
        conditioning: Optional[Conditioning] = None,
    ):
        if conditioning is None:
            conditioning = self.conditioning()

        predicted_noise = self.unet(
            diffused_latents, self.indices(from_indices), conditioning.encodings
        )["sample"]
        return predicted_noise.float()

    def forward(self, diffused_latents, indices, conditioning=None):
        indices = self.indices(indices)
        return Predictions(
            from_diffused_latents=diffused_latents,
            from_indices=indices,
            predicted_noise=self.predicted_noise(
                diffused_latents, indices, conditioning
            ),
            schedule_alphas=self.schedule_alphas,
            schedule_sigmas=self.schedule_sigmas,
        )

    def predictions(self, diffused_latents, ts, conditioning=None):
        return self.forward(diffused_latents, ts, conditioning)

    def conditioning(self, texts=None, images=None, encodings=None):
        """
        Create a conditioning object from a list of texts. Unconditional is an empty string.
        """
        if texts is None and images is None and encodings is None:
            texts = [""]

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
        text_encodings = text_encoder(tokenized_text.input_ids.to(self.device))[0]
        return Conditioning(encodings=text_encodings)

    def diffuse_latents(self, denoised_latents, indices, noise=None):
        indices = self.indices(indices)
        if noise is None:
            noise = torch.randn_like(denoised_latents)
        alphas, sigmas = self.alphas(indices), self.sigmas(indices)
        return denoised_latents * alphas + noise * sigmas


StableDiffusion: Model = cache(Model)


def test_stable_diffusion_step():
    from diffusers import DDIMScheduler

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    model_name = "CompVis/stable-diffusion-v1-4"
    batch_size = 1
    height = 512
    width = 512

    from_index = 999
    to_index = 998

    texts = ["painting of a dog"]

    diffusion_model = Model(model_name).to(device)

    diffused_latents = diffusion_model.random_diffused_latents((1, 3, 512, 512))

    conditioning = diffusion_model.conditioning(texts)
    predictions = diffusion_model.predictions(
        diffused_latents, from_index, conditioning
    )
    compare_next_diffused_latents = predictions.step(to_index)

    del diffusion_model

    # compare with diffusers
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
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
    assert torch.allclose(
        next_diffused_latents, compare_next_diffused_latents, atol=1e-3
    )


def test_stable_diffusion():
    from tqdm import tqdm
    from perceptor import utils

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    diffusion_model = StableDiffusion().to(device)
    conditioning = diffusion_model.conditioning(["photograph of a playful cat"])
    unconditioned_conditioning = diffusion_model.conditioning()
    diffused_latents = diffusion_model.random_diffused_latents((1, 3, 512, 512)).to(
        device
    )

    progress_bar = tqdm(diffusion_model.schedule_indices(to_index=0, n_steps=50))
    for from_indices, to_indices in progress_bar:

        for _ in range(10):
            positive_predictions = diffusion_model.predictions(
                diffused_latents,
                from_indices,
                conditioning,
            )
            diffused_latents = positive_predictions.step(to_indices)

            unconditioned_predictions = diffusion_model.predictions(
                diffused_latents,
                to_indices,
                unconditioned_conditioning,
            )
            if from_indices.item() <= 700:
                diffused_latents = unconditioned_predictions.noisy_reverse_step(
                    from_indices
                )
            else:
                diffused_latents = unconditioned_predictions.reverse_step(from_indices)

        step_predictions = diffusion_model.predictions(
            diffused_latents,
            from_indices,
            conditioning,
        )
        diffused_latents = step_predictions.step(to_indices)

        utils.pil_image(
            diffusion_model.images(step_predictions.denoised_latents).clamp(0, 1)
        ).save("tests/stable_diffusion.png")

        progress_bar.set_postfix(
            dict(
                from_indices=from_indices.item(),
                to_indices=to_indices.item(),
            )
        )

    predictions = diffusion_model.predictions(
        diffused_latents,
        to_indices,
        conditioning,
    )

    utils.pil_image(
        diffusion_model.images(predictions.denoised_latents).clamp(0, 1)
    ).save("tests/stable_diffusion.png")


def test_stable_diffusion_init_image():
    import requests
    from PIL import Image
    import torch
    import torchvision.transforms.functional as TF
    from tqdm import tqdm
    from perceptor import utils

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    init_image = TF.to_tensor(
        Image.open(requests.get(image_url, stream=True).raw).resize((512, 512))
    )[None].cuda()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    from_index = 400

    diffusion_model = StableDiffusion().to(device)
    conditioning = diffusion_model.conditioning(["photograph of playful lions"])
    unconditioned_conditioning = diffusion_model.conditioning()
    diffused_latents = diffusion_model.diffuse_latents(
        diffusion_model.latents(init_image), from_index
    )

    for from_indices, to_indices in tqdm(
        diffusion_model.schedule_indices(from_index=from_index, to_index=0, n_steps=50)
    ):
        for _ in range(4):
            # upscaled guided forward, random backward
            unconditioned_predictions = diffusion_model.predictions(
                diffused_latents,
                from_indices,
                unconditioned_conditioning,
            )
            positive_predictions = diffusion_model.predictions(
                diffused_latents,
                from_indices,
                conditioning,
            )
            guided_predictions = unconditioned_predictions.classifier_free_guidance(
                positive_predictions, guidance_scale=7.0
            )
            diffused_latents = guided_predictions.resample(to_indices)

        predictions = diffusion_model.predictions(
            diffused_latents,
            from_indices,
            conditioning,
        )
        diffused_latents = predictions.step(to_indices)

    predictions = diffusion_model.predictions(
        diffused_latents,
        to_indices,
        conditioning,
    )

    utils.pil_image(
        diffusion_model.images(predictions.denoised_latents).clamp(0, 1)
    ).save("tests/stable_diffusion_init_image.png")
