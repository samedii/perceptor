import torch
import torch.nn.functional as F
from torchvision import transforms
from basicsr.utils.download_util import load_file_from_url

from perceptor import utils
from .blip_itm import blip_itm


checkpoints = {
    "model_base_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
    "model_large_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth",
    "model_base_retrieval_flickr": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
    "model_large_retrieval_flickr": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth",
    "model*_base_caption": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth",
    "model_large_caption": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
    "model_vqa": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth",
    "model*_vqa": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth",
    "model_base_nlvr": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth",
    "model_large": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "model*_base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base.pth",
    "model_base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
}

vit_map = {
    "model_base_retrieval_coco": "base",
    "model_large_retrieval_coco": "large",
    "model_base_retrieval_flickr": "base",
    "model_large_retrieval_flickr": "large",
    "model*_base_caption": "base",
    "model_large_caption": "large",
    "model_vqa": "base",
    "model*_vqa": "base",
    "model_base_nlvr": "base",
    "model_large": "large",
    "model*_base": "base",
    "model_base": "base",
}


@utils.cache
class BLIP(torch.nn.Module):
    def __init__(self, name="model_base_retrieval_flickr"):
        super().__init__()
        self.name = name

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        self.image_size = 384
        self.model = (
            blip_itm(
                pretrained=checkpoint_path,
                image_size=self.image_size,
                vit=vit_map[self.name],
            )
            .requires_grad_(False)
            .eval()
        )

        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def encode_texts(self, texts):
        tokenized_texts = self.model.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )

        text_output_itc = self.model.text_encoder(
            tokenized_texts.input_ids,
            attention_mask=tokenized_texts.attention_mask,
            return_dict=True,
            mode="text",
        )

        text_encodings_itc = F.normalize(
            self.model.text_proj(text_output_itc.last_hidden_state[:, 0, :]),
            dim=-1,
        )

        return tokenized_texts, text_encodings_itc

    def encode_images(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
            )

        image_embeddings = self.model.visual_encoder(self.normalize(images))

        image_encodings_itc = F.normalize(
            self.model.vision_proj(image_embeddings[:, 0, :]), dim=-1
        )

        return image_embeddings, image_encodings_itc

    def image_text_contrastive_spherical_distance(self, encodings_a, encodings_b):
        return (
            (encodings_a[None, :] - encodings_b[:, None])
            .norm(dim=-1)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )

    def image_text_retrieval_probabilities(self, tokenized_texts, image_embeddings):
        n_texts = len(tokenized_texts.input_ids)
        n_images = len(image_embeddings)
        device = image_embeddings.device

        image_attentions = torch.ones(
            image_embeddings.size()[:-1], dtype=torch.long
        ).to(device)

        text_output_itm = self.model.text_encoder(
            tokenized_texts.input_ids.to(device).repeat(n_images, 1),
            attention_mask=tokenized_texts.attention_mask.repeat(n_images, 1).to(
                device
            ),
            encoder_hidden_states=image_embeddings.repeat(n_texts, 1, 1),
            encoder_attention_mask=image_attentions.repeat(n_texts, 1),
            return_dict=True,
        )
        return (
           F.softmax(  # softmax in original. optimizing logit gives it a huge strength
                self.model.itm_head(
                    text_output_itm.last_hidden_state[:, 0, :].to(device)
                ),
                dim=1,
            )[:, 1]
        )  # fmt: skip

    def forward(self, _):
        raise NotImplementedError
