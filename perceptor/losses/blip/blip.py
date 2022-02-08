import torch
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

        self.text = self.model.tokenizer(
            self.text_prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )

        text_output_itc = self.model.text_encoder(
            self.text.input_ids,
            attention_mask=self.text.attention_mask,
            return_dict=True,
            mode="text",
        )

        self.text_features = torch.nn.Parameter(
            F.normalize(
                self.model.text_proj(text_output_itc.last_hidden_state[:, 0, :]),
                dim=-1,
            ),
            requires_grad=False,
        )

    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
            )

        image_embeds = self.model.visual_encoder(self.normalize(images))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device
        )

        text_output_itm = self.model.text_encoder(
            self.text.input_ids.to(images.device).repeat(len(images), 1),
            attention_mask=self.text.attention_mask.repeat(len(images), 1).to(
                images.device
            ),
            encoder_hidden_states=image_embeds.repeat(len(self.text_prompts), 1, 1),
            encoder_attention_mask=image_atts.repeat(len(self.text_prompts), 1),
            return_dict=True,
        )
        itm_loss = -F.softmax(  # softmax in original. optimizing logit gives it a huge strength
            self.model.itm_head(
                text_output_itm.last_hidden_state[:, 0, :].to(images.device)
            ),
            dim=1,
        )[
            :, 1
        ].mean()

        image_features = F.normalize(
            self.model.vision_proj(image_embeds[:, 0, :]), dim=-1
        )

        spherical_distance_itc = (
            (image_features[None, :] - self.text_features[:, None])
            .norm(dim=-1)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        ).mean()
        return (spherical_distance_itc + itm_loss) / 2
