import torch
import torch.nn.functional as F

from perceptor import utils
from .load import load


@utils.cache
class RuCLIP(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model, self.clip_processor = load(name, start_device)
        self.model.requires_grad_(False).eval()

        # softmax on cpu does not support half precision
        if not torch.cuda.is_available():
            self.model.float()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def encode_texts(self, text_prompts):
        inputs = self.clip_processor(
            text=text_prompts, return_tensors="pt", padding=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        return F.normalize(self.model.encode_text(input_ids))

    def encode_images(self, images):
        return F.normalize(
            self.model.encode_image(
                self.clip_processor.image_transform(images.to(self.device))
            )
        )

    def forward(self, _):
        raise NotImplementedError
