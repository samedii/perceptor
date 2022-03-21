# https://github.com/CompVis/latent-diffusion/blob/main/scripts/sample_diffusion.py

sr_diffMode = "superresolution"
sr_model = get_model("superresolution")


def download_models(mode):

    if mode == "superresolution":
        # this is the small bsr light model
        url_conf = 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'
        url_ckpt = 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'

        path_conf = f'{model_path}/superres/project.yaml'
        path_ckpt = f'{model_path}/superres/last.ckpt'

        download_url(url_conf, path_conf)
        download_url(url_ckpt, path_ckpt)

        path_conf = path_conf + '/?dl=1' # fix it
        path_ckpt = path_ckpt + '/?dl=1' # fix it
        return path_conf, path_ckpt

    else:
        raise NotImplementedError

def get_model(mode):
    path_conf, path_ckpt = download_models(mode)
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model


def do_superres(img, filepath):

    if args.sharpen_preset == "Faster":
        sr_diffusion_steps = "25"
        sr_pre_downsample = "1/2"
    if args.sharpen_preset == "Fast":
        sr_diffusion_steps = "100"
        sr_pre_downsample = "1/2"
    if args.sharpen_preset == "Slow":
        sr_diffusion_steps = "25"
        sr_pre_downsample = "None"
    if args.sharpen_preset == "Very Slow":
        sr_diffusion_steps = "100"
        sr_pre_downsample = "None"

    sr_post_downsample = "Original Size"
    sr_diffusion_steps = int(sr_diffusion_steps)
    sr_eta = 1.0
    sr_downsample_method = "Lanczos"

    gc.collect()
    torch.cuda.empty_cache()

    im_og = img
    width_og, height_og = im_og.size

    # Downsample Pre
    if sr_pre_downsample == "1/2":
        downsample_rate = 2
    elif sr_pre_downsample == "1/4":
        downsample_rate = 4
    else:
        downsample_rate = 1

    width_downsampled_pre = width_og // downsample_rate
    height_downsampled_pre = height_og // downsample_rate

    if downsample_rate != 1:
        # print(f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
        im_og = im_og.resize(
            (width_downsampled_pre, height_downsampled_pre), Image.LANCZOS
        )
        # im_og.save('/content/temp.png')
        # filepath = '/content/temp.png'

    logs = sr_run(sr_model["model"], im_og, sr_diffMode, sr_diffusion_steps, sr_eta)

    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1.0, 1.0)
    sample = (sample + 1.0) / 2.0 * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    a = Image.fromarray(sample[0])

    # Downsample Post
    if sr_post_downsample == "1/2":
        downsample_rate = 2
    elif sr_post_downsample == "1/4":
        downsample_rate = 4
    else:
        downsample_rate = 1

    width, height = a.size
    width_downsampled_post = width // downsample_rate
    height_downsampled_post = height // downsample_rate

    if sr_downsample_method == "Lanczos":
        aliasing = Image.LANCZOS
    else:
        aliasing = Image.NEAREST

    if downsample_rate != 1:
        # print(f'Downsampling from [{width}, {height}] to [{width_downsampled_post}, {height_downsampled_post}]')
        a = a.resize((width_downsampled_post, height_downsampled_post), aliasing)
    elif sr_post_downsample == "Original Size":
        # print(f'Downsampling from [{width}, {height}] to Original Size [{width_og}, {height_og}]')
        a = a.resize((width_og, height_og), aliasing)

    display.display(a)
    a.save(filepath)
    return
    print(f"Processing finished!")
