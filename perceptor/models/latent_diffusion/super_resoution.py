from PIL import Image
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision.datasets.utils import download_url
import einops

from .ldm.util import ismap
from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.util import instantiate_from_config


def super_resolution(image: Image.Image, steps=100):
    """Super resolution x4 using latent diffusion. Not differentiable"""
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

    @torch.no_grad()
    def convsample_ddim(
        model,
        cond,
        steps,
        shape,
        eta=1.0,
        callback=None,
        normals_sequence=None,
        mask=None,
        x0=None,
        quantize_x0=False,
        img_callback=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        x_T=None,
        log_every_t=None,
    ):

        ddim = DDIMSampler(model)
        bs = shape[0]  # dont know where this comes from but wayne
        shape = shape[1:]  # cut batch dim
        print(f"Sampling with eta = {eta}; steps: {steps}")
        samples, intermediates = ddim.sample(
            steps,
            batch_size=bs,
            shape=shape,
            conditioning=cond,
            callback=callback,
            normals_sequence=normals_sequence,
            quantize_x0=quantize_x0,
            eta=eta,
            mask=mask,
            x0=x0,
            temperature=temperature,
            verbose=False,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
        )

        return samples, intermediates

    @torch.no_grad()
    def make_convolutional_sample(
        batch,
        model,
        mode="vanilla",
        custom_steps=None,
        eta=1.0,
        swap_mode=False,
        masked=False,
        invert_mask=True,
        quantize_x0=False,
        custom_schedule=None,
        decode_interval=1000,
        resize_enabled=False,
        custom_shape=None,
        temperature=1.0,
        noise_dropout=0.0,
        corrector=None,
        corrector_kwargs=None,
        x_T=None,
        save_intermediate_vid=False,
        make_progrow=True,
        ddim_use_x0_pred=False,
    ):
        log = dict()

        z, c, x, xrec, xc = model.get_input(
            batch,
            model.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=not (
                hasattr(model, "split_input_params")
                and model.cond_stage_key == "coordinates_bbox"
            ),
            return_original_cond=True,
        )
        print("z.shape", z.shape, "c.shape", c.shape, "x.shape", x.shape)

        log_every_t = 1 if save_intermediate_vid else None

        if custom_shape is not None:
            z = torch.randn(custom_shape)
            print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

        z0 = None

        log["input"] = x
        log["reconstruction"] = xrec

        if ismap(xc):
            log["original_conditioning"] = model.to_rgb(xc)
            if hasattr(model, "cond_stage_key"):
                log[model.cond_stage_key] = model.to_rgb(xc)

        else:
            log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_model:
                log[model.cond_stage_key] = (
                    xc if xc is not None else torch.zeros_like(x)
                )
                if model.cond_stage_key == "class_label":
                    log[model.cond_stage_key] = xc[model.cond_stage_key]

        with model.ema_scope("Plotting"):
            img_cb = None

            sample, intermediates = convsample_ddim(
                model,
                c,
                steps=custom_steps,
                shape=z.shape,
                eta=eta,
                quantize_x0=quantize_x0,
                img_callback=img_cb,
                mask=None,
                x0=z0,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=corrector,
                corrector_kwargs=corrector_kwargs,
                x_T=x_T,
                log_every_t=log_every_t,
            )

            if ddim_use_x0_pred:
                sample = intermediates["pred_x0"][-1]

        x_sample = model.decode_first_stage(sample)

        try:
            x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
            log["sample_noquant"] = x_sample_noquant
            log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
        except:
            pass

        log["sample"] = x_sample
        log["time"] = 0

        return log

    example = dict()
    up_f = 4

    c = image
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c_up = torchvision.transforms.functional.resize(
        c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True
    )
    c_up = einops.rearrange(c_up, "1 c h w -> 1 h w c")
    c = einops.rearrange(c, "1 c h w -> 1 h w c")
    c = 2.0 * c - 1.0

    c = c.to(torch.device("cuda"))
    example["LR_image"] = c
    example["image"] = c_up

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = "ddim"
    ddim_use_x0_pred = False
    temperature = 1.0
    eta = 1.0
    make_progrow = True
    custom_shape = None
    custom_steps = steps
    resize_enabled = False

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

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

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(
                model.device
            )
            x_T = einops.repeat(x_T, "1 c h w -> b c h w", b=custom_shape[0])

        logs = make_convolutional_sample(
            example,
            model,
            mode=mode,
            custom_steps=custom_steps,
            eta=eta,
            swap_mode=False,
            masked=masked,
            invert_mask=invert_mask,
            quantize_x0=False,
            custom_schedule=None,
            decode_interval=10,
            resize_enabled=resize_enabled,
            custom_shape=custom_shape,
            temperature=temperature,
            noise_dropout=0.0,
            corrector=guider,
            corrector_kwargs=ckwargs,
            x_T=x_T,
            save_intermediate_vid=save_intermediate_vid,
            make_progrow=make_progrow,
            ddim_use_x0_pred=ddim_use_x0_pred,
        )

    return logs["sample"].add(1).div(2).clamp(0, 1)
