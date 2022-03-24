# TODO: there are probably still some issues with the implementation and converting
# t to the space of the original t
import math
import torch
from torch import nn
from basicsr.utils.download_util import load_file_from_url

from .unet import UNetModel


class GuidedDiffusion(nn.Module):
    def __init__(self, multiply_t=1000 / 701):
        """
        Args:
            multiply_t: heuristic multiplier that handles how much noies the model
                expects. The default value is 1000 / 701 and was selected based on
                the initially picked value being 701 and the total steps when the
                model was trained is 1000. It has also been experimentally verified
                to give decent results.
        """
        super().__init__()
        self.multiply_t = multiply_t
        self.shape = (3, 512, 512)
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

        self.model = create_model(**model_config)

        checkpoint_path = load_file_from_url(
            "https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt",
            "models",
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.requires_grad_(False).eval()

        # from disco diffusion, might not be needed
        # for name, param in self.model.named_parameters():
        #     if "qkv" in name or "norm" in name or "proj" in name:
        #         param.requires_grad_()
        if model_config["use_fp16"]:
            self.model.convert_to_fp16()

        original_betas = torch.linspace(0.0001, 0.02, 1000)
        original_alphas = 1.0 - original_betas
        self.original_alphas_cumprod = nn.Parameter(
            torch.cumprod(original_alphas, dim=0), requires_grad=False
        )

    def forward(self, x, t):
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"Guided diffusion model only works well with shape {self.shape}"
            )

        alpha, sigma = t_to_alpha_sigma(t)
        _, closest_indices = (
            (self.original_alphas_cumprod[None, :] - alpha[:, None])
            .abs()
            .topk(2, largest=False)
        )
        closest = self.original_alphas_cumprod[closest_indices]
        original_t = (
            torch.lerp(
                closest_indices[:, 0].float(),
                closest_indices[:, 1].float(),
                (alpha - closest[:, 0]) / (closest[:, 1] - closest[:, 0]),
            )
            * self.multiply_t
        )
        eps = self.model(x, 1 + original_t)[:, :3]
        velocity = (eps - x * sigma[:, None, None, None]) / alpha[:, None, None, None]
        return velocity


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


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
