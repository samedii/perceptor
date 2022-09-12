import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

from perceptor import utils
from perceptor.transforms.resize import resize


@utils.cache
class OpenCLIP(torch.nn.Module):
    def __init__(
        self, architecture="ViT-B-32", weights="laion2b_e16", precision=None, jit=False
    ):
        """
        Args:
            architecture (str): name of the clip model
            weights (str): name of the weights

            Available weight/model combinations are (in order of relevance):
            - ("ViT-B-32", "laion2b_e16") (65.7%)
            - ("ViT-B-16-plus-240", "laion400m_e32") (69.2%)
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

        weights_path = open_clip.pretrained.download_pretrained(
            open_clip.pretrained.get_pretrained_url(architecture, weights),
            root="models",
        )

        # softmax on cpu does not support half precision
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if precision is None:
            if start_device == torch.device("cuda"):
                precision = "fp16"
            else:
                precision = "fp32"

        if weights == "openai":
            self.model = open_clip.load_openai_model(
                weights_path, start_device, jit=jit
            ).eval()
            if precision == "fp32":
                model = model.float()
        else:
            self.model = open_clip.create_model(
                architecture,
                weights_path,
                device=start_device,
                precision=precision,
                jit=jit,
            ).eval()

        if jit is False:
            self.model = self.model.requires_grad_(False)

        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

    def to(self, device):
        if device == torch.device("cpu"):
            self.model = self.model.float()
        return super().to(device)

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
                    out_shape=(
                        self.model.visual.image_size,
                        self.model.visual.image_size,
                    ),
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

    model = OpenCLIP("ViT-B-32", "laion2b_e16")

    image = torch.randn((1, 3, 256, 256)).requires_grad_()
    with torch.enable_grad():
        model.encode_images(image).mean().backward()

    assert image.grad is not None
