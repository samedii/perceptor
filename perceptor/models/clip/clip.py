import torch
import torch.nn.functional as F

from perceptor import utils
from .clip_base import get_clip_perceptor


@utils.cache
class CLIP(torch.nn.Module):
    def __init__(self, name):
        """
        Args:
            name: name of the clip model. Available models are:
                - RN50
                - RN101
                - RN50x4
                - RN50x16
                - RN50x64
                - ViT-B/32
                - ViT-B/16
                - ViT-L/14
        """

        super().__init__()
        self.name = name
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = get_clip_perceptor(name, start_device).requires_grad_(False).eval()

        # softmax on cpu does not support half precision
        if not torch.cuda.is_available():
            self.model.float()

    def encode_texts(self, text_prompts):
        return F.normalize(self.model.encode_text(text_prompts))

    def encode_images(self, images):
        return F.normalize(self.model.encode_image(images.to(self.model.device)))

    def forward(self, _):
        raise NotImplementedError
