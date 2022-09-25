from .unet import UNetModel
from .script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


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
