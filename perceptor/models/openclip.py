import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

from perceptor import utils
from perceptor.transforms.resize import resize


@utils.cache
class OpenCLIP(torch.nn.Module):
    def __init__(self, archicture="ViT-B-32-quickgelu", weights="laion400m_e31"):
        super().__init__()
        self.archicture = archicture
        self.weights = weights

        # softmax on cpu does not support half precision
        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        precision = start_device == torch.device("cuda")
        self.model = open_clip.create_model(
            archicture, weights, device=start_device, precision=precision
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
        # return torch.cat(
        #     [
        #         F.normalize(self.model.encode_text(text_prompt))
        #         for text_prompt in text_prompts
        #     ]
        # )
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
