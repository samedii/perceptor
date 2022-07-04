import torch
from torch import nn
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from . import diffusion_space, utils
from .unet import UNetModel
from .smaller_diffusion_model import SmallerDiffusionModel


@cache
class GuidedDiffusion(nn.Module):
    def __init__(self, name="standard"):
        """
        Args:
            name: The name of the model.
        """
        super().__init__()
        self.name = name

        if name == "standard":
            self.model = create_openimages_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt",
                # alternative: "https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_openimages_epoch28_withfilter.pt",
                "models",
            )
            self.shape = (3, 512, 512)
        elif name == "smaller":
            self.model = SmallerDiffusionModel()
            checkpoint_path = load_file_from_url(
                "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth",
                "models",
            )
            self.shape = (3, 512, 512)
        elif name == "pixelart":
            self.model = create_pixelart_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt",
                "models",
            )
            self.shape = (3, 256, 256)
        else:
            raise ValueError(f"Unknown model name {self.name}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.requires_grad_(False).eval()

    def forward(self, diffused, t):
        return self.denoise(diffused, t)

    def velocity(self, diffused, t):
        x = diffusion_space.encode(diffused)
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"Guided diffusion model only works well with shape {self.shape}"
            )

        if isinstance(t, int) or isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)

        if self.name == "standard":
            return self.model(x, t * 1000)[:, :3]
        elif self.name == "smaller":
            return self.model(x, t)
        elif self.name == "pixelart":
            return self.model(x, t * 1000)[:, :3]
        else:
            raise ValueError(f"Unknown model name {self.name}")

    def alphas(self, t):
        if t.ndim == 0:
            t = t[None]
        alphas, _ = utils.t_to_alpha_sigma(t)
        return alphas[:, None, None, None]

    def sigmas(self, t):
        if t.ndim == 0:
            t = t[None]
        _, sigmas = utils.t_to_alpha_sigma(t)
        return sigmas[:, None, None, None]

    @staticmethod
    def x(denoised, noise, t):
        pred = diffusion_space.encode(denoised)
        if isinstance(t, int) or isinstance(t, float) or t.ndim == 0:
            t = torch.full((pred.shape[0],), t).to(pred)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return diffusion_space.decode(
            pred * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]
        )

    def denoise(self, diffused, t, eps=None):
        x = diffusion_space.encode(diffused)
        if isinstance(t, int) or isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)

        alphas, sigmas = utils.t_to_alpha_sigma(t)
        if eps is None:
            eps = self.eps(diffused, t)

        return diffusion_space.decode(
            x * alphas[:, None, None, None]
            - (eps - x * sigmas[:, None, None, None])
            * sigmas[:, None, None, None]
            / alphas[:, None, None, None]
        )

    @staticmethod
    def diffuse(images, t, noise=None):
        x0 = diffusion_space.encode(images)
        if isinstance(t, int) or isinstance(t, float) or t.ndim == 0:
            t = torch.full((x0.shape[0],), t).to(x0)
        if noise is None:
            noise = torch.randn_like(x0)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return diffusion_space.decode(
            x0 * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]
        )

    def eps(self, diffused, t):
        x = diffusion_space.encode(diffused)
        if isinstance(t, int) or isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t).to(x)
        velocity = self.velocity(diffused, t)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        return x * sigmas[:, None, None, None] + velocity * alphas[:, None, None, None]

    def step(self, from_diffused, denoised, from_t, to_t, eta=None):
        from_x = diffusion_space.encode(from_diffused)
        pred = diffusion_space.encode(denoised)

        from_alphas, from_sigmas = self.alphas(from_t), self.sigmas(from_t)
        to_alphas, to_sigmas = self.alphas(to_t), self.sigmas(to_t)

        velocity = (from_x * from_alphas - pred) / from_sigmas
        eps = from_x * from_sigmas + velocity * from_alphas

        if eta is not None:
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
            to_x += torch.randn_like(to_x) * ddim_sigma
        else:
            to_x = pred * to_alphas + eps * to_sigmas

        return diffusion_space.decode(to_x)


def create_openimages_model():
    model_config = dict(
        image_size=512,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32, 16, 8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        learn_sigma=True,
    )

    model = create_model(**model_config)
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model


def create_pixelart_model():
    model_config = dict(
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

    model = create_model(**model_config)
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model


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
