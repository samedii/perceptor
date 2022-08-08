import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

from perceptor import utils
from perceptor.transforms.resize import resize


@utils.cache
class OpenCLIP(torch.nn.Module):
    def __init__(self, archicture="ViT-B-32", weights="laion2b_e16"):
        """
        Args:
            archicture (str): name of the clip model
            weights (str): name of the weights

            Available weight/model combinations are (in order of relevance):
            - ("ViT-B-32", "laion2b_e16") (65.62%)
            - ("ViT-B-16-plus-240", "laion400m_e32") (69.21%)
            - ("ViT-B-16", "laion400m_e32") (67.07%)
            - ("ViT-B-32", "laion400m_e32") (62.96%)
            - ("ViT-L-14", "laion400m_e32") (72.77%)
            - ("RN101", "yfcc15m") (34.8%)
            - ("RN50", "yfcc15m") (32.7%)
            - ("RN50", "cc12m") (36.45%)
            - ("RN50-quickgelu", "openai")
            - ("RN101-quickgelu", "openai")
            - ("RN50x4", "openai")
            - ("RN50x16", "openai")
            - ("RN50x64", "openai")
            - ("ViT-B-32-quickgelu", "openai")
            - ("ViT-B-16", "openai")
            - ("ViT-L-14", "openai")
            - ("ViT-L-14-336", "openai")
        """
        super().__init__()
        self.archicture = archicture
        self.weights = weights

        # softmax on cpu does not support half precision
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        precision = start_device == torch.device("cuda")
        self.model = (
            open_clip.create_model(
                archicture, weights, device=start_device, precision=precision
            )
            .eval()
            .requires_grad_(False)
        )

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

    def encode_texts(self, text_prompts):
        return F.normalize(
            self.model.encode_text(open_clip.tokenize(text_prompts).to(self.device))
        )

    def encode_images(self, images):
        return F.normalize(
            self.model.encode_image(
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
        )

    def forward(self, _):
        raise NotImplementedError
