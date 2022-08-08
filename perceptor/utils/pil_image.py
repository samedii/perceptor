from PIL import Image
from torchvision.transforms.functional import to_pil_image
from lantern import Tensor


def pil_image(images: Tensor) -> Image:
    if images.max() > 1 or images.min() < 0:
        print("Warning: images are not in range [0, 1]")
    n, c, h, w = images.shape
    return to_pil_image(
        images.permute(0, 2, 3, 1).reshape(-1, w, c).permute(2, 0, 1).clamp(0, 1)
    )
