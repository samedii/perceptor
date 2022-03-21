"""
Work-in-progress code for noise lens. Unclear if this will work so dropping it for now.
"""
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision.datasets.utils import download_url
import einops

from perceptor.models.latent_diffusion.ldm.util import instantiate_from_config, default


@torch.no_grad()
def noise_lens(images, t, noise=None):
    """
    Add noise to an image and then try to remove it.

    Args:
        image (torch.Tensor): Input image tensor NCHW (0-1).
        t (int): noise timestep between 0 and 1000 where 0 is no noise.
        noise (torch.Tensor): Optional noise tensor.
    """
    url_conf = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
    url_ckpt = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"

    path_conf = f"sharpen-colab.yaml"
    path_ckpt = f"sharpen-colab.ckpt"

    download_url(url=url_conf, root="models", filename=path_conf)
    download_url(url=url_ckpt, root="models", filename=path_ckpt)

    sd = torch.load(f"models/{path_ckpt}", map_location="cpu")["state_dict"]

    config = OmegaConf.load(f"models/{path_conf}")
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    example = dict()
    up_f = 4

    c = images
    c_up = torchvision.transforms.functional.resize(
        c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True
    )
    c_up = einops.rearrange(c_up, "1 c h w -> 1 h w c")
    c = einops.rearrange(c, "1 c h w -> 1 h w c")
    c = 2.0 * c - 1.0

    c = c.to(torch.device("cuda"))
    example = dict(
        LR_image=c,
        image=c_up,
    )
    z, c = model.get_input(
        example,
        model.first_stage_key,
        force_c_encode=not (
            hasattr(model, "split_input_params")
            and model.cond_stage_key == "coordinates_bbox"
        ),
    )

    x = example["image"].permute(0, 3, 1, 2).cuda()
    c = x
    t = torch.tensor([t]).cuda()
    if model.model.conditioning_key is not None:
        assert c is not None
        if model.cond_stage_trainable:
            c = model.get_learned_conditioning(c)

    x_start = x
    cond = c

    noise = default(noise, lambda: torch.randn_like(x_start))
    x_noisy = model.q_sample(x_start=x_start, t=t, noise=noise)

    if model.parameterization != "eps":
        raise NotImplementedError()

    # pred_x0 = model.p_mean_variance(
    #     x_noisy,
    #     cond,
    #     t,
    #     clip_denoised=True,
    #     quantize_denoised=False,
    #     return_x0=True,
    # )[-1]

    samples = model.p_sample(
        x_noisy,
        cond,
        t,
        # clip_denoised=True,
        # quantize_denoised=False,
        return_x0=True,
    )[-1]

    # print(pred_x0.shape)
    return model.decode_first_stage(samples)
