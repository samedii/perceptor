from torchvision.transforms.functional import to_pil_image


def pil_image(images):
    n, c, h, w = images.shape
    return to_pil_image(images.permute(0, 2, 3, 1).reshape(-1, w, c).permute(2, 0, 1))
