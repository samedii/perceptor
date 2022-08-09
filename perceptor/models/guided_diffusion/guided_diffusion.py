import torch
from torch import nn
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from . import diffusion_space
from .unet import UNetModel
from .script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


@cache
class GuidedDiffusion(nn.Module):
    def __init__(self, name="standard"):
        """
        Args:
            name: The name of the model.

        Usage:

            diffusion = models.GuidedDiffusion("pixelart").to(device)

            diffused_image = torch.randn((1, 3, 256, 256))

            for from_index, to_index in model.schedule_indices():
                eps = diffusion.eps(diffused_image, from_index)
                denoised_image = diffusion.denoise(diffused_image, from_index, eps)
                diffused_image = diffusion.step(diffused_image, eps, from_index, to_index)
            denoised_image = diffusion.denoise(diffused_image, to_index)
        """
        super().__init__()
        self.name = name

        if name == "standard":
            self.model, self.diffusion = create_openimages_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt",
                # alternative: "https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_openimages_epoch28_withfilter.pt",
                "models",
            )
            self.shape = (3, 512, 512)
        elif name == "pixelart":
            self.model, self.diffusion = create_pixelart_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt",
                "models",
            )
            self.shape = (3, 256, 256)
        else:
            raise ValueError(f"Unknown model name {self.name}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.requires_grad_(False).eval()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def schedule_indices(self, from_index=999, to_index=20, n_steps=None):
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

    def random_diffused(self, shape):
        return diffusion_space.decode(torch.randn(shape)).to(self.device)

    def forward(self, diffused, from_index):
        return self.denoise(diffused, from_index)

    def denoise(self, diffused, from_index, eps=None):
        x = diffusion_space.encode(diffused)
        if isinstance(from_index, int) or from_index.ndim == 0:
            from_index = torch.full((x.shape[0],), from_index).to(x.device)

        if eps is None:
            eps = self.eps(diffused, from_index)

        return diffusion_space.decode(
            self.diffusion._predict_xstart_from_eps(x, from_index, eps)
        )

    def diffuse(self, images, to_index, noise=None):
        x0 = diffusion_space.encode(images)
        if isinstance(to_index, int) or to_index.ndim == 0:
            to_index = torch.full((x0.shape[0],), to_index).to(x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        return diffusion_space.decode(self.diffusion.q_sample(x0, to_index, noise))

    def eps(self, diffused, from_index):
        x = diffusion_space.encode(diffused)
        if isinstance(from_index, int) or from_index.ndim == 0:
            from_index = torch.full((x.shape[0],), from_index).to(x.device)
        return self.model(x, from_index)[:, :3]

    def guided_eps(self, eps, grad, from_index, guidance_scale=1000):
        """
        Guided eps by differentiating through the diffusion model.

        Usage:

            diffused_image.requires_grad_(True)
            with torch.enable_grad():
                eps = diffusion.eps(diffused_image, from_index)
                denoised_image = diffusion.denoise(diffused_image, from_index, eps)

                augmentations_ = torch.cat([augmentations(denoised_image) for _ in range(4)])
                torch.stack(
                    [
                        (text_loss(augmentations_) / len(text_losses))
                        for text_loss in text_losses
                    ]
                ).mean().backward()

            eps = diffusion.guided_eps(eps, diffused_image.grad, from_index)
        """
        return (
            eps + guidance_scale * (1 - self.alphas_cumprod(from_index)).sqrt() * grad
        )

    def ts(self, index):
        return torch.tensor([index], device=self.device, dtype=torch.float32)

    def alphas_cumprod(self, index):
        return (
            torch.as_tensor(self.diffusion.alphas_cumprod[index])
            .float()
            .to(self.device)[None, None, None, None]
        )

    def sqrt_one_minus_alphas_cumprod(self, index):
        return (
            torch.as_tensor(self.diffusion.sqrt_one_minus_alphas_cumprod[index])
            .float()
            .to(self.device)[None, None, None, None]
        )

    def step(self, from_diffused, eps, from_index, to_index, noise=None, eta=1.0):
        if to_index > from_index:
            raise ValueError("to_index must be smaller than from_index")
        if noise is None:
            noise = torch.randn_like(eps)

        pred = diffusion_space.encode(self.denoise(from_diffused, from_index, eps))

        from_alphas_cumprod = self.alphas_cumprod(from_index)
        to_alphas_cumprod = self.alphas_cumprod(to_index)

        to_sigmas = eta * torch.sqrt(
            (1 - to_alphas_cumprod)
            / (1 - from_alphas_cumprod)
            * (1 - from_alphas_cumprod / to_alphas_cumprod)
        )

        dir_xt = (1.0 - to_alphas_cumprod - to_sigmas**2).sqrt() * eps
        to_x = to_alphas_cumprod.sqrt() * pred + dir_xt + to_sigmas * noise
        return diffusion_space.decode(to_x)


def create_openimages_model():
    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            # 'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            # 'rescale_timesteps': True,
            # 'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": True,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )

    model, diffusion = create_model_and_diffusion(**model_config)
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def create_pixelart_model():
    model_config = model_and_diffusion_defaults()
    model_config.update(
        dict(
            image_size=256,
            learn_sigma=True,
            num_channels=128,
            num_res_blocks=2,
            num_heads=1,
            num_heads_upsample=-1,
            num_head_channels=-1,
            attention_resolutions="16",
            channel_mult="",
            dropout=0.0,
            class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_fp16=True,
            use_new_attention_order=False,
        )
    )

    model, diffusion = create_model_and_diffusion(**model_config)
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def test_pixelart_diffusion():
    from perceptor import utils

    model = GuidedDiffusion("pixelart").cuda()
    diffused = model.random_diffused((1, 3, 256, 256))

    for from_index, to_index in model.schedule_indices(n_steps=50):
        eps = model.eps(diffused, from_index)
        diffused = model.step(diffused, eps, from_index, to_index)
    denoised = model.denoise(diffused, to_index)
    utils.pil_image(denoised).save("tests/pixelart.png")


def test_guided_diffusion_diffusion():
    from perceptor import utils

    model = GuidedDiffusion("standard").cuda()
    diffused = model.random_diffused((1, 3, 512, 512))

    for from_index, to_index in model.schedule_indices(n_steps=50):
        eps = model.eps(diffused, from_index)
        diffused = model.step(diffused, eps, from_index, to_index)
    denoised = model.denoise(diffused, to_index)
    utils.pil_image(denoised).save("tests/guided_diffusion.png")
