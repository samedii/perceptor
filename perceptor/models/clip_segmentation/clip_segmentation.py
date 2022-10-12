import torch
import torch.nn.functional as F
import torchvision.transforms
from .models.clipseg import CLIPDensePredT

from perceptor import utils
from perceptor.transforms.resize import resize

checkpoint_urls = {
    "rd16-uni": "https://s3.eu-central-1.wasabisys.com/nextml-model-data/clip-seg/rd16-uni.pth",
    "rd64-uni-refined": "https://s3.eu-central-1.wasabisys.com/nextml-model-data/clip-seg/rd64-uni-refined.pth",
    "rd64-uni": "https://s3.eu-central-1.wasabisys.com/nextml-model-data/clip-seg/rd64-uni.pth",
}


@utils.cache
class CLIPSegmentation(torch.nn.Module):
    def __init__(self, name="rd64-uni-refined"):
        super().__init__()

        if name == "rd64-uni-refined":
            model = CLIPDensePredT(
                version="ViT-B/16", reduce_dim=64, complex_trans_conv=True
            )
            model.load_state_dict(
                torch.load("models/rd64-uni-refined.pth"), strict=False
            )
        elif name == "rd64-uni":
            self.model = (
                CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
                .eval()
                .requires_grad_(False)
            )
            self.model.load_state_dict(
                torch.load("models/rd64-uni.pth", map_location="cpu"),
                strict=False,
            )
        self.size = (352, 352)
        self.normalize = (
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @torch.cuda.amp.autocast()
    def encode_texts(self, text_prompts, normalize=True):
        encodings = self.model.encode_text(
            open_clip.tokenize(text_prompts).to(self.device)
        )
        if normalize:
            return F.normalize(encodings)
        else:
            return encodings

    @torch.cuda.amp.autocast()
    def encode_images(self, images, normalize=True):
        encodings = self.model.encode_image(
            self.normalize(
                resize(
                    images.to(self.device),
                    out_shape=self.image_size,
                )
            )
        )

        if normalize:
            return F.normalize(encodings)
        else:
            return encodings

    def forward(self, _):
        raise NotImplementedError


def test_open_clip():
    import torch

    model = OpenCLIP("ViT-B-32", "laion2b_s34b_b79k")

    image = torch.randn((1, 3, 256, 256)).requires_grad_()
    with torch.enable_grad():
        model.encode_images(image).mean().backward()

    assert image.grad is not None
