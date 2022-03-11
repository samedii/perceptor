import torch
import torch.nn.functional as F

from perceptor import utils
from .slip_base import get_slip_perceptor


@utils.cache
class SLIP(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.model = get_slip_perceptor(name)
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def device(self):
        return next(iter(self.parameters()))

    def encode_texts(self, text_prompts):
        return F.normalize(self.model.encode_text(text_prompts))

    def encode_images(self, images):
        return F.normalize(self.model.encode_image(images.to(self.device)))

    def forward(self, _):
        raise NotImplementedError
