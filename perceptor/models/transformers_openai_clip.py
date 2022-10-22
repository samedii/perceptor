from dataclasses import dataclass
import lantern
import torch
import torchvision.transforms
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPFeatureExtractor,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from perceptor import utils
from perceptor.transforms.resize import resize


@dataclass
class Encodings:
    features: BaseModelOutputWithPooling
    unnormalized_encodings: lantern.Tensor
    encodings: lantern.Tensor


# @utils.cache
class TransformersOpenAICLIP(torch.nn.Module):
    def __init__(
        self,
        name="openai/clip-vit-large-patch14",
        bfloat16=True,
    ):
        """
        CLIP text-image similarity. Text model is only loaded on demand.
        Slower than `OpenCLIP` implementation but has easy feature extraction.

        Args:
            name (str): huggingface model id or path to weights
                Available weight/model combinations are (in order of relevance):
                - laion/CLIP-ViT-H-14-laion2B-s32B-b79K (78.0%)
                - laion/CLIP-ViT-g-14-laion2B-s12B-b42K (76.6%)
                - laion/CLIP-ViT-L-14-laion2B-s32B-b82K (75.3%)
                - laion/CLIP-ViT-B-32-laion2B-s34B-b79K (66.6%)
                - openai/clip-vit-base-patch32 (63.3%)
                - openai/clip-vit-base-patch16 (68.3%)
                - openai/clip-vit-large-patch14 (75.6%)
                - openai/clip-vit-large-patch14-336 (76.6%)
                - M-CLIP/XLM-Roberta-Large-Vit-B-16Plus (95.0% COCO@10)
                - M-CLIP/XLM-Roberta-Large-Vit-L-14 (92.4% COCO@10)
                - M-CLIP/XLM-Roberta-Large-Vit-B-32 (91.8% COCO@10)
                - M-CLIP/LABSE-Vit-L-14 (91.6% COCO@10)
                - sentence-transformers/clip-ViT-B-32-multilingual-v1
                - Huggingface model id
                - Local weights
            bfloat16 (bool): use bfloat16 for inference
        """
        super().__init__()
        self.name = name

        clip_model = (
            CLIPModel.from_pretrained(name, use_bfloat16=bfloat16)
            .eval()
            .requires_grad_(False)
        )

        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale

        vision_feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.image_size = [vision_feature_extractor.size for _ in range(2)]
        self.normalize = torchvision.transforms.Normalize(
            vision_feature_extractor.image_mean,
            vision_feature_extractor.image_std,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def tokenize(self, texts):
        tokenizer = CLIPTokenizer.from_pretrained(self.name)
        return tokenizer(texts, padding=True, return_tensors="pt")

    def encode_texts(self, texts) -> Encodings:
        inputs = self.tokenize(texts)
        text_model = CLIPTextModel.from_pretrained(self.name).to(self.device)
        features = text_model(
            **{key: value.to(self.device) for key, value in inputs.items()}
        )

        unnormalized_encodings = self.text_projection(features.pooler_output)

        return Encodings(
            features=features,
            unnormalized_encodings=unnormalized_encodings,
            encodings=unnormalized_encodings
            / unnormalized_encodings.norm(p=2, dim=-1, keepdim=True),
        )

    def encode_images(self, images) -> Encodings:
        features = self.vision_model(
            self.normalize(
                resize(
                    images.to(self.device),
                    out_shape=self.image_size,
                )
            )
        )
        unnormalized_encodings = self.visual_projection(features.pooler_output)

        return Encodings(
            features=features,
            unnormalized_encodings=unnormalized_encodings,
            encodings=unnormalized_encodings
            / unnormalized_encodings.norm(p=2, dim=-1, keepdim=True),
        )

    @staticmethod
    def spherical_distance(
        encodings_a: Encodings,
        encodings_b: Encodings,
    ) -> lantern.Tensor:
        return (
            (encodings_a.encodings[:, None] - encodings_b.encodings[None, :])
            .norm(dim=2)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
        )

    def forward(self, _):
        raise NotImplementedError


def test_transformers_clip_gradients():
    import torch

    model = TransformersOpenAICLIP().cuda()

    image = torch.randn((1, 3, 256, 256)).requires_grad_()
    text_encodings = model.encode_texts(["a dog", "a cat"])

    with torch.enable_grad():
        image_encoding = model.encode_images(image)
        model.spherical_distance(text_encodings, image_encoding).mean().backward()

    assert image.grad is not None


def test_transformers_clip_same():
    import torch
    from perceptor.models.open_clip import OpenCLIP

    torch.set_grad_enabled(False)

    open_clip = OpenCLIP("ViT-L-14", "openai").cuda()

    image = torch.rand((1, 3, 256, 256))
    reference = open_clip.encode_images(image)

    model = TransformersOpenAICLIP("openai/clip-vit-large-patch14").cuda()
    image_encoding = model.encode_images(image)

    assert (image_encoding.encodings - reference).abs().max() <= 1e-3
