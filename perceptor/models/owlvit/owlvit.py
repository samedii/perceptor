from typing import List
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from lantern import Tensor, FunctionalBase
from transformers import OwlViTProcessor
from transformers.tokenization_utils_base import BatchEncoding

from .modeling_owlvit import OwlViTForObjectDetection
from .feaure_extraction_owlvit import OwlViTFeatureExtractor
from perceptor import utils
from perceptor.transforms.resize import resize


class OWLViTEncodings(nn.Module):
    texts: List[List[str]]
    batch_encoding: BatchEncoding

    def __init__(self, texts, batch_encoding):
        super().__init__()
        self.texts = texts
        self.batch_encoding = batch_encoding

    def to(self, device):
        self.batch_encoding.to(device)
        return super().to(device)

    def repeat(self, repetitions):
        return OWLViTEncodings(
            texts=self.texts,
            batch_encoding=BatchEncoding(
                {
                    key: torch.cat([value for _ in range(repetitions)], dim=0)
                    for key, value in self.batch_encoding.items()
                }
            ),
        )


class OWLViTPredictions(FunctionalBase):
    logits: Tensor.dims("NKE")
    boxes: Tensor.dims("NK4")
    scores: Tensor.dims("NK")
    labels: Tensor.dims("NK")
    texts: List[List[str]]


@utils.cache
class OWLViT(torch.nn.Module):
    def __init__(self):
        """
        OWL-ViT zero-shot text-conditioned bounding box model
        """
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.processor.feature_extractor = OwlViTFeatureExtractor()
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.size = (768, 768)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def encode_texts(self, texts: List[List[str]]) -> OWLViTEncodings:
        try:
            processed = self.processor(text=texts, return_tensors="pt")
        except ValueError as e:
            raise ValueError(
                "Failed to encode texts, they are probably too long"
            ) from e
        return OWLViTEncodings(
            texts=texts,
            batch_encoding=processed.to(self.device),
        )

    def forward(
        self, images: Tensor.dims("NCHW"), encodings: OWLViTEncodings
    ) -> OWLViTPredictions:
        if images.shape[-2:] != self.size:
            images = resize(images, out_shape=self.size)

        images = TF.normalize(
            images,
            mean=self.processor.feature_extractor.image_mean,
            std=self.processor.feature_extractor.image_std,
        )

        inputs = BatchEncoding(
            data=dict(
                pixel_values=images,
                **encodings.repeat(images.shape[0]).batch_encoding,
            ),
            tensor_type="pt",
        )

        outputs = self.model(**inputs)
        target_sizes = torch.as_tensor([images.shape[-2:]]).repeat(len(images), 1)
        results = self.processor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )

        boxes, scores, labels = zip(
            *[
                (result["boxes"], result["scores"], result["labels"])
                for result in results
            ]
        )

        return OWLViTPredictions(
            logits=outputs.logits,
            boxes=torch.stack(boxes),
            scores=torch.stack(scores),
            labels=torch.stack(labels),
            texts=encodings.texts,
        )


def test_owlvit():
    import requests
    from PIL import Image
    import torchvision.transforms.functional as TF
    import transformers

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    images = (
        TF.to_tensor(Image.open(requests.get(image_url, stream=True).raw))[None]
        .cuda()
        .requires_grad_()
    )
    texts = [["a photo of a cat", "a photo of a dog"]]

    model = OWLViT().cuda()
    encodings = model.encode_texts(texts)

    with torch.enable_grad():
        # print(encodings.batch_encoding.device)
        predictions = model(images, encodings)
        predictions.scores.mean().backward()

    assert images.grad is not None

    images = resize(images, out_shape=(768, 768))
    images = TF.to_pil_image(images[0])

    reference_processor = transformers.OwlViTProcessor.from_pretrained(
        "google/owlvit-base-patch32"
    )
    inputs = reference_processor(text=texts, images=images, return_tensors="pt")

    reference_model = transformers.OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    )
    outputs = reference_model(**inputs)
    target_sizes = torch.as_tensor([images.size[::-1]])
    results = reference_processor.post_process(
        outputs=outputs, target_sizes=target_sizes
    )

    boxes, scores = (
        results[0]["boxes"],
        results[0]["scores"],
    )

    assert torch.allclose(boxes[0], predictions.boxes[0, 0].cpu(), atol=10)
    assert torch.allclose(scores[0], predictions.scores[0, 0].cpu(), atol=1e-2)


def test_owlvit_too_long_text():
    import pytest

    text_prompts = [
        "photo of a warrior. trending on artstation. #photorealistic #artstation",
        "photo of a strong handsome warrior",
    ]
    with pytest.raises(ValueError):
        OWLViT().encode_texts(text_prompts)
