import torch.nn.functional as F
from torchvision import transforms
from basicsr.utils.download_util import load_file_from_url

from perceptor.losses.interface import LossInterface
from .blip_itm import blip_itm


checkpoints = {
    "model_base_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
}


class BLIP(LossInterface):
    def __init__(self, text_prompts, name="model_base_retrieval_coco"):
        super().__init__()
        self.name = name
        self.text_prompts = text_prompts

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        self.image_size = 384
        self.model = blip_itm(
            pretrained=checkpoint_path, image_size=self.image_size, vit="base"
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, image):
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
            )

        return -sum(
            [
                F.softmax(
                    self.model(
                        self.normalize(image),
                        text_prompt,
                        match_head="itm",
                    ),
                    dim=1,
                )[:, 1].mean()
                for text_prompt in self.text_prompts
            ]
        ) / len(self.text_prompts)
