from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


resume_specs = {
    # For StyleGAN2/ADA models; --cfg=stylegan2
    "stylegan2": {
        # Official NVIDIA models
        "ffhq256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl",
        "ffhqu256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl",
        "ffhq512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl",
        "ffhq1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl",
        "ffhqu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl",
        "celebahq256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl",
        "lsundog256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl",
        "afhqcat512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl",
        "afhqdog512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl",
        "afhqwild512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl",
        "afhq512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl",
        "brecahad512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl",
        "cifar10": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl",
        "metfaces1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl",
        "metfacesu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl",
        # Community models; TODO: add the interesting ones found in https://github.com/justinpinkney/awesome-pretrained-stylegan2
        "minecraft1024": "https://github.com/jeffheaton/pretrained-gan-minecraft/releases/download/v1/minecraft-gan-2020-12-22.pkl",  # Thanks to @jeffheaton
        # Deceive-D/APA models (ignoring the faces models): https://github.com/EndlessSora/DeceiveD
        "afhqcat256": "https://drive.google.com/u/0/uc?export=download&confirm=zFoN&id=1P9ouHIK-W8JTb6bvecfBe4c_3w6gmMJK",
        "anime256": "https://drive.google.com/u/0/uc?export=download&confirm=6Uie&id=1EWOdieqELYmd2xRxUR4gnx7G10YI5dyP",
        "cub256": "https://drive.google.com/u/0/uc?export=download&confirm=KwZS&id=1J0qactT55ofAvzddDE_xnJEY8s3vbo1_",
    },
    # For StyleGAN3 config-r models (--cfg=stylegan3-r)
    "stylegan3-r": {
        # Official NVIDIA models
        "afhq512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl",
        "ffhq1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl",
        "ffhqu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl",
        "ffhqu256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl",
        "metfaces1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl",
        "metfacesu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl",
    },
    # For StyleGAN3 config-t models (--cfg=stylegan3-t)
    "stylegan3-t": {
        # Official NVIDIA models
        "afhq512": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl",
        "ffhq1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl",
        "ffhqu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl",
        "ffhqu256": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl",
        "metfaces1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl",
        "metfacesu1024": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl",
        # Community models, found in: https://github.com/justinpinkney/awesome-pretrained-stylegan3 by @justinpinkney
        "landscapes256": "https://drive.google.com/u/0/uc?export=download&confirm=eJHe&id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1",  # Thanks to @justinpinkney
        "wikiart1024": "https://drive.google.com/u/0/uc?export=download&confirm=2tz5&id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj",  # Thanks to @justinpinkney
    },
}


def z_to_img(
    G,
    latents: torch.Tensor,
    label: torch.Tensor,
    truncation_psi: float,
    noise_mode: str = "const",
) -> np.ndarray:
    """
    Get an image/np.ndarray from a latent Z using G, the label, truncation_psi, and noise_mode. The shape
    of the output image/np.ndarray will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels]
    """
    assert isinstance(
        latents, torch.Tensor
    ), f'latents should be a torch.Tensor!: "{type(latents)}"'
    if len(latents.shape) == 1:
        latents = latents.unsqueeze(0)  # An individual latent => [1, G.z_dim]
    img = G(z=latents, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    return img


def w_to_img(
    G,
    dlatents: Union[List[torch.Tensor], torch.Tensor],
    noise_mode: str = "const",
) -> np.ndarray:
    """
    Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
    returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
    """
    assert isinstance(
        dlatents, torch.Tensor
    ), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
    if len(dlatents.shape) == 2:
        dlatents = dlatents.unsqueeze(
            0
        )  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
    synth_image = G.synthesis(dlatents, noise_mode=noise_mode)
    return synth_image


def get_w_from_seed(
    G,
    batch_sz: int,
    device: torch.device,
    truncation_psi: float,
    seed: Optional[int],
    # centroids_path: Optional[str],
    class_idx: Optional[int],
) -> torch.Tensor:
    """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional)"""

    if G.c_dim != 0:
        # sample random labels if no class idx is given
        if class_idx is None:
            class_indices = np.random.RandomState(seed).randint(
                low=0, high=G.c_dim, size=(batch_sz)
            )
            class_indices = torch.from_numpy(class_indices).to(device).long()
            w_avg = G.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
            class_indices = torch.full((batch_sz,), class_idx).to(device)

        labels = F.one_hot(class_indices, G.c_dim)

    else:
        w_avg = G.mapping.w_avg.unsqueeze(0)
        labels = None
        if class_idx is not None:
            print("Warning: --class is ignored when running an unconditional network")

    z = np.random.RandomState(seed).randn(batch_sz, G.z_dim)
    z = torch.from_numpy(z).to(device)
    w = G.mapping(z, labels)

    w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    w = w_avg + (w - w_avg) * truncation_psi

    return w
