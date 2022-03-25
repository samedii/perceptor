import torch
import torch.nn.functional as F

from perceptor import utils
from .slip_base import get_slip_perceptor


@utils.cache
class SLIP(torch.nn.Module):
    def __init__(self, name):
        """
        Args:
            name: name of the slip model. Available models are:
                - SLIP_VITS16
                - SLIP_VITB16
                - SLIP_VITL16
                - CLIP_VITS16
                - CLIP_VITB16
                - CLIP_VITL16
                - SLIP_CC3M
                - SLIP_CC12M
        """
        super().__init__()
        self.name = name
        self.model = get_slip_perceptor(name).requires_grad_(False).eval()

    @property
    def device(self):
        return next(iter(self.parameters()))

    def encode_texts(self, text_prompts):
        return F.normalize(self.model.encode_text(text_prompts))

    def encode_images(self, images):
        return F.normalize(self.model.encode_image(images.to(self.device)))

    def forward(self, _):
        raise NotImplementedError
