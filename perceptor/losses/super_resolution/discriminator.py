import torch
from basicsr.utils.download_util import load_file_from_url

from perceptor.losses.interface import LossInterface
from .unet_discriminator_sn import UNetDiscriminatorSN


checkpoints = {
    "RealESRGAN_x4plus_netD": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth",
}


class SuperResolutionDiscriminator(LossInterface):
    def __init__(self, name="RealESRGAN_x4plus_netD"):
        super().__init__()
        self.name = name

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        self.model = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)
        self.model.requires_grad_(False)
        weights = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(weights["params"], strict=True)

    def forward(self, images):
        return -self.model(images).mean() * 0.001
