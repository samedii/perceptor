from omegaconf import OmegaConf
import torch
import torchvision
from torchvision.datasets.utils import download_url
import einops

from perceptor import transforms
from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.util import instantiate_from_config


def noise_lens(image, t, noise=None):
    """
    Super resolution x4 using latent diffusion

    Args:
        image (torch.Tensor): tensor of shape (1, 3, H, W)
        t (float): timestep (0-1)
        noise (torch.Tensor): tensor of shape (1, 3, H, W)
    """
    steps = 100
    t = t * steps
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

    c = image
    c_up = torchvision.transforms.functional.resize(
        c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True
    )
    c_up = einops.rearrange(c_up, "1 c h w -> 1 h w c")
    c = einops.rearrange(c, "1 c h w -> 1 h w c")
    c = 2.0 * c - 1.0

    c = c.to(torch.device("cuda"))
    example["LR_image"] = c
    example["image"] = c_up
    print("c.shape", c.shape, "c_up.shape", c_up.shape)

    temperature = 1.0
    eta = 1.0

    _, height, width, _ = example["image"].shape
    print("height", height, "width", width)
    split_input = height > 128 and width > 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {
            "ks": (ks, ks),
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
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    z, c = model.get_input(
        example,
        model.first_stage_key,
        return_first_stage_outputs=False,
        force_c_encode=not (
            hasattr(model, "split_input_params")
            and model.cond_stage_key == "coordinates_bbox"
        ),
        return_original_cond=False,
    )

    with model.ema_scope("Plotting"):

        ddim = DDIMSampler(model)
        bs = z.shape[0]
        shape = z.shape[1:]

        ddim.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
        C, H, W = shape

        index = (
            (torch.from_numpy(ddim.ddim_timesteps).to(image.device) - t)
            .abs()
            .argmin()
            .to(image.device)
        )
        t = ddim.ddim_timesteps[index]
        ts = torch.full((bs,), t, device=c.device, dtype=torch.long)

        img = model.q_sample(x_start=z, t=ts, noise=noise)

        img, pred_x0 = ddim.p_sample_ddim(
            img,
            c,
            ts,
            index,
            # quantize_denoised=True,
            temperature=temperature,
        )

    x_sample = model.decode_first_stage(pred_x0)

    return transforms.clamp_with_grad(x_sample, 0, 1)
