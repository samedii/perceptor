import torch
import torch.nn.functional as F

from perceptor import utils
from perceptor.transforms.resize import resize
from . import model_pt, pretrained


models = {
    "16-epochs": "cloob_laion_400m_vit_b_16_16_epochs",
    "32-epochs": "cloob_laion_400m_vit_b_16_32_epochs",
}


@utils.cache
class CLOOB(torch.nn.Module):
    def __init__(self, name="16-epochs"):
        """
        Args:
            name: name of the cloob model. Available models are:
                - 16-epochs
                - 32-epochs
        """
        super().__init__()
        self.name = name
        config = pretrained.get_config(models[self.name])
        image_size = config["image_encoder"]["image_size"]
        self.image_size = (image_size, image_size)
        self.model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        self.model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
        self.model.eval().requires_grad_(False)

    @property
    def device(self):
        return next(iter(self.parameters()))

    def encode_texts(self, text_prompts):
        return self.model.text_encoder(self.model.tokenize(text_prompts))

    def encode_images(self, images):
        if tuple(images.shape[-2:]) != self.image_size:
            images = resize(
                images,
                out_shape=self.image_size,
            )
        return F.normalize(
            self.model.image_encoder(
                self.model.normalize(images.to(self.device))
            ).float(),
            dim=-1,
        )

    def forward(self, _):
        raise NotImplementedError
