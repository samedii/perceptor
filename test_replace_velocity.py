# %%
from IPython.display import display, clear_output
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia.augmentation as K

from perceptor import init, drawers, transforms, losses, utils, models

seed = np.random.randint(100000)
# seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

image_size = (512, 512)


device = torch.device("cuda")

text_prompts = ["warrior. concept art. trending on artstation"]
# text_prompts = ["photograph of a sunset at the beach"]
# text_prompts = ["a blueprint of a steampunk submarine"]

# init_images = init.fractal((1, 3, 256, 256))
init_images = TF.to_tensor(Image.open("lion.jpg").resize(image_size))[None].to(device)

# drawers.Diffusion.from_image(image=init_images, init_noise=0.5)

diffusion = (
    models.VelocityDiffusion(
        name="cc12m_1_cfg",
    )
    .add_texts_(text_prompts)
    .to(device)
)


background_mask = (
    (
        TF.to_tensor(Image.open("human_shape.png").convert("RGB").resize(image_size))[
            None
        ].to(device)
        == 1
    )
    .all(dim=1, keepdim=True)
    .float()
)

crop_size = (450, 450)

# try augmentations without resizing?

augmentations = torch.nn.Sequential(
    K.RandomCrop(size=crop_size),
    # K.RandomResizedCrop(
    #     size=crop_size,
    #     scale=(0.7, 0.95),
    #     ratio=(0.85, 1.2),
    #     cropping_mode="resample",
    #     p=1.0,
    # ),
    K.ColorJitter(hue=0.025, saturation=0.05, p=0.8),
)

target_images = TF.to_tensor(Image.open("human.jpg"))[None]

text_losses = [
    # losses.BLIP(name="model_base_retrieval_flickr"),
    # losses.BLIP(name="model_large_retrieval_flickr"),
    losses.CLIP(name="ViT-B/16"),
    losses.CLIP(name="ViT-B/32"),
    # losses.CLIP(name="RN50x16"),
    # losses.CLIP(name="RN50x4"),
]
for text_loss in text_losses:
    text_loss.add_texts_(text_prompts).add_images_(target_images).to(device)

aesthetic_loss = losses.Aesthetic().to(device)


def guide(images, state):
    images = transforms.clamp_with_grad(images, 0, 1)

    augmentations_ = torch.cat([augmentations(images) for _ in range(8)])

    for text_loss in text_losses:
        (text_loss(augmentations_) / len(text_losses) * 1).backward(retain_graph=True)

    # (
    #     background_mask.mul(transforms.clamp_with_grad(images, -100, 0.95).sub(0.95))
    #     .square()
    #     .mean()
    #     .mul(10.0)
    #     .backward(retain_graph=True)
    # )

    # (
    #     (1 - background_mask)
    #     .mul(transforms.clamp_with_grad(images, 0.8, 100).sub(0.8))
    #     .square()
    #     .mean()
    #     .mul(80.0)
    #     .backward(retain_graph=True)
    # )

    (aesthetic_loss(images) * 0.01).backward()
    cond_grad = -state.x.grad

    guided_v = (
        state.v.detach()
        - cond_grad
        * (state.sigmas[:, None, None, None] / state.alphas[:, None, None, None])
        * 500
    )

    # return state.v.detach() - cond_grad * 50 * (
    #     state.sigmas[:, None, None, None] / state.alphas[:, None, None, None]
    # )

    guided_pred = (
        state.x * state.alphas[:, None, None, None]
        - guided_v * state.sigmas[:, None, None, None]
    )
    guided_images = (guided_pred + 1) / 2

    alpha_mask = background_mask * (1 - state.alphas) ** 2
    wanted_images = guided_images * (1 - alpha_mask) + 1 * alpha_mask
    # alpha_mask = (1 - background_mask) * (1 - state.alphas)
    alpha_mask = 1 - background_mask
    wanted_images = (
        wanted_images * (1 - alpha_mask)
        + wanted_images.clip(max=state.alphas * 0.3 + 0.7) * alpha_mask
    )
    wanted_pred = wanted_images * 2 - 1

    wanted_v = (
        state.x * state.alphas[:, None, None, None] - wanted_pred
    ) / state.sigmas[:, None, None, None]
    return wanted_v


n_iterations = 100

for iteration, images in tqdm(
    enumerate(
        diffusion.inverse(
            (torch.randn_like(init_images) + 1) / 2,
            from_noise=1.0,
            to_noise=0.0,
            n_steps=n_iterations,
            guide=guide,
        )
    )
):

    if iteration == 0 or (iteration + 1) % 5 == 0 or iteration == n_iterations - 1:
        display_image = utils.pil_image(images.clamp(0, 1))
        display_image.save("output.png", compress=False)
        # utils.pil_image(augmentations_).save("augmentations.png", compress=False)
        clear_output(wait=True)
        display(display_image)


# %%


utils.pil_image((diffusion_drawer.latent + 1) / 2)
# %%

utils.pil_image(small_images_)
# %%

next(small_image.parameters()).shape
# %%
