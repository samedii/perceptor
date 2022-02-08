import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from .clip_base import get_clip_perceptor


class CLIP(LossInterface):
    def __init__(self, text_prompts, name="ViT-B/16"):
        super().__init__()
        self.name = name
        self.text_prompts = text_prompts

        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = get_clip_perceptor(name, start_device)
        self.model.eval()
        self.model.requires_grad_(False)

        # softmax on cpu does not support half precision
        if not torch.cuda.is_available():
            self.model.float()

        self.text_encodings = torch.nn.Parameter(
            self.model.encode_text(text_prompts), requires_grad=False
        )

    def forward(self, images):
        image_encodings = self.model.encode_image(images).float()

        input_normed = F.normalize(image_encodings.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.text_encodings.unsqueeze(0), dim=2)
        dists = (
            (input_normed - embed_normed).norm(dim=2).div(2).arcsin().square().mul(2)
        )
        return dists.mean()
