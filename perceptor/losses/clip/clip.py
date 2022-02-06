import torch
import torch.nn.functional as F

from perceptor.losses.interface import LossInterface
from .clip_base import get_clip_perceptor


def parse_prompt(prompt):
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals) :]
    return vals[0], float(vals[1]), float(vals[2])


class CLIP(LossInterface):
    def __init__(self, text_prompts, name="ViT-B/16"):
        super().__init__()
        self.name = name
        self.text_prompts = text_prompts

        self.model = get_clip_perceptor(name, torch.device("cuda"))
        self.model.eval()
        self.model.requires_grad_(False)

        self.text_encodings = self.model.encode_text(text_prompts[0])

    def forward(self, images):
        image_encodings = self.model.encode_image(images).float()

        input_normed = F.normalize(image_encodings.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.text_encodings.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        return dists.mean()
