import torch
import torch.nn.functional as F
from basicsr.utils.download_util import load_file_from_url

from .model_creation import create_clip_model
from perceptor import utils


CHECKPOINT_URLS = {
    "clip/image-enc": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/clip_image_enc.pt",
    "clip/text-enc": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/clip_text_enc.pt",
}


@utils.cache
class GlideCLIP(torch.nn.Module):
    def __init__(self):
        """CLIP model trained to handle noisy images"""
        super().__init__()

        start_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = create_clip_model(device=start_device).eval().requires_grad_(False)
        self.model.image_encoder.load_state_dict(
            torch.load(
                load_file_from_url(CHECKPOINT_URLS["clip/image-enc"], "models"),
                map_location=start_device,
            )
        )
        self.model.text_encoder.load_state_dict(
            torch.load(
                load_file_from_url(CHECKPOINT_URLS["clip/text-enc"], "models"),
                map_location=start_device,
            )
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def encode_texts(self, text_prompts):
        return F.normalize(self.model.text_embeddings(text_prompts))

    def encode_images(self, diffused, ts):
        """
        Args:
            diffused: [batch_size, 3, 64, 64] diffused images between 0 and 1
            t: [batch_size] timestep between 0 and 999 where 0 is without noise.
        """
        return F.normalize(
            self.model.image_embeddings(
                diffused.to(self.device).mul(2).sub(1),
                ts.to(self.device),
            )
        )

    def forward(self, _):
        raise NotImplementedError
