from torchvision.transforms.functional import to_pil_image


def pil_image(images):
    return to_pil_image(
        images.permute(0, 2, 3, 1).reshape(-1, images.shape[-1], 3).permute(2, 0, 1)
    )
