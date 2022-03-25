import torch
import torch.nn.functional as F
from torchvision import transforms
from basicsr.utils.download_util import load_file_from_url

from perceptor import utils
from perceptor.transforms.resize import resize
from .blip_itm import blip_itm


checkpoints = {
    "model_base_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
    "model_large_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth",
    "model_base_retrieval_flickr": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
    "model_large_retrieval_flickr": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth",
    "model_large": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "model*_base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base.pth",
    "model_base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
    "model_base_capfilt_large": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth",
}

vit_map = {
    "model_base_retrieval_coco": "base",
    "model_large_retrieval_coco": "large",
    "model_base_retrieval_flickr": "base",
    "model_large_retrieval_flickr": "large",
    "model_large": "large",
    "model*_base": "base",
    "model_base": "base",
    "model_base_capfilt_large": "base",
}

image_size = {
    "model_base_retrieval_coco": 384,
    "model_large_retrieval_coco": 384,
    "model_base_retrieval_flickr": 384,
    "model_large_retrieval_flickr": 384,
    "model_large": 384,
    "model*_base": 384,
    "model_base": 224,
    "model_base_capfilt_large": 384,
}


@utils.cache
class BLIP(torch.nn.Module):
    def __init__(self, name="model_base_retrieval_flickr"):
        """
        Args:
            name (str): Name of the model. Available models are:
                - model_base_retrieval_coco
                - model_large_retrieval_coco
                - model_base_retrieval_flickr
                - model_large_retrieval_flickr
                - model_large
                - model*_base
                - model_base
                - model_base_capfilt_large
        """
        super().__init__()
        self.name = name

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        self.image_size = image_size[name]
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
        text_output = self.model.text_encoder(
            tokenized_texts.input_ids,
            attention_mask=tokenized_texts.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_encodings = F.normalize(
            self.model.text_proj(text_output.last_hidden_state[:, 0, :]),
            dim=-1,
        )
        return F.normalize(text_encodings, dim=-1)

    def encode_images(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = resize(images, out_shape=(self.image_size, self.image_size))

        image_encodings = self.model.visual_encoder(self.normalize(images))

        image_encodings_itc = F.normalize(
            self.model.vision_proj(image_encodings[:, 0, :]), dim=-1
        )

        return F.normalize(image_encodings_itc, dim=-1)

    def image_text_contrastive_spherical_distance(self, encodings_a, encodings_b):
        return (
            (encodings_a[None, :] - encodings_b[:, None])
            .norm(dim=-1)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )

    def forward(self, _):
        raise NotImplementedError
