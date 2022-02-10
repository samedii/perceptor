import torch
import torch.nn.functional as F

from perceptor import utils
from .clip_base import get_clip_perceptor


@utils.cache
class CLIP(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = get_clip_perceptor(name, start_device)
        self.model.eval()
        self.model.requires_grad_(False)

        # softmax on cpu does not support half precision
        if not torch.cuda.is_available():
            self.model.float()

    def encode_texts(self, text_prompts):
        return F.normalize(self.model.encode_text(text_prompts))

    def encode_images(self, images):
        return F.normalize(self.model.encode_image(images.to(self.model.device)))

    def forward(self, _):
        raise NotImplementedError
