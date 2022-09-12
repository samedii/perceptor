from typing import Optional

from .open_clip import OpenCLIP


def CLIP(architecture: str, precision: Optional[str] = None, jit=True):
    """
    Args:
        architecture (str): name of the clip model. Available models are:
            - RN50 [-quickgelu]
            - RN101 [-quickgelu]
            - RN50x4
            - RN50x16
            - RN50x64
            - ViT-B-32 [-quickgelu]
            - ViT-B-16
            - ViT-L-14
            - ViT-L-14-336px
        precision (str): precision of the model. Options are "fp32" and "fp16"
    """
    if "-quickgelu" not in architecture and architecture in [
        "RN50",
        "RN101",
        "ViT-B-32",
    ]:
        architecture = architecture + "-quickgelu"
    return OpenCLIP(architecture, "openai", precision, jit=jit)


def test_clip():
    import torch

    model = CLIP("ViT-B-32")

    image = torch.randn((1, 3, 256, 256)).requires_grad_()
    with torch.enable_grad():
        model.encode_images(image).mean().backward()

    assert image.grad is not None
