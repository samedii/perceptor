import torch
import torch.nn.functional as F
import torchvision.transforms
import open_clip

from perceptor import utils
from perceptor.transforms.resize import resize


@utils.cache
class OpenCLIP(torch.nn.Module):
    def __init__(
        self,
        architecture="ViT-H-14",
        weights="laion2b_s32b_b79k",
        precision=None,
        jit=False,
    ):
        """
        Args:
            architecture (str): name of the clip model
            weights (str): name of the weights

            Available weight/model combinations are (in order of relevance):
            - ("ViT-H-14", "laion2b_s32b_b79k") (78.0%)
            - ("ViT-g-14", "laion2b_s12b_b42k") (76.6%)
            - ("ViT-L-14", "laion2b_s32b_b82k") (75.3%)
            - ("ViT-B-32", "laion2b_s34b_b79k") (66.6%)
            - ("ViT-B-16-plus-240", "laion400m_e32") (69.2%)
            - ("ViT-B-32", "laion2b_e16") (65.7%)
            - ("ViT-B-16", "laion400m_e32") (67.0%)
            - ("ViT-B-32", "laion400m_e32") (62.9%)
            - ("ViT-L-14", "laion400m_e32") (72.8%)
            - ("RN101", "yfcc15m") (34.8%)
            - ("RN50", "yfcc15m") (32.7%)
            - ("RN50", "cc12m") (36.45%)
            - ("RN50-quickgelu", "openai") (59.6%)
            - ("RN101-quickgelu", "openai")
            - ("RN50x4", "openai")
            - ("RN50x16", "openai")
            - ("RN50x64", "openai")
            - ("ViT-B-32-quickgelu", "openai") (63.3%)
            - ("ViT-B-16", "openai") (68.3%)
            - ("ViT-L-14", "openai") (75.6%)
            - ("ViT-L-14-336", "openai") (76.6%)
        """
        super().__init__()
        self.architecture = architecture
        self.weights = weights

        if (architecture, weights) not in open_clip.list_pretrained():
            raise ValueError(f"Invalid architecture/weights: {architecture}/{weights}")

        # softmax on cpu does not support half precision
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if precision is None:
            if start_device == torch.device("cuda"):
                precision = "fp16"
            else:
                precision = "fp32"

        self.model, _, transforms = open_clip.create_model_and_transforms(
            architecture,
            weights,
            device=start_device,
            precision=precision,
            jit=jit,
            cache_dir="models",
        )
        self.model = self.model.eval()

        if jit is False:
            self.model = self.model.requires_grad_(False)

        self.normalize = torchvision.transforms.Normalize(
            transforms.transforms[-1].mean,
            transforms.transforms[-1].std,
        )

    def to(self, device):
        if device == torch.device("cpu"):
            self.model = self.model.float()
        return super().to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def image_size(self):
        if isinstance(self.model.visual.image_size, tuple):
            return self.model.visual.image_size
        else:
            return (self.model.visual.image_size, self.model.visual.image_size)

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
