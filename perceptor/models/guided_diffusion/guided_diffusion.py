from typing import Optional
from contextlib import contextmanager
import torch
import lantern
from basicsr.utils.download_util import load_file_from_url

from perceptor.utils import cache
from . import diffusion_space
from .create_models import create_openimages_model, create_pixelart_model
from .predictions import Predictions


# @cache
class GuidedDiffusion(torch.nn.Module):
    def __init__(self, name="standard"):
        """
        Args:
            name: The name of the model. Available models are "standard" and "pixelart"
        """
        super().__init__()
        self.name = name

        if name == "standard":
            self.model, self.scheduler = create_openimages_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt",
                # alternative: "https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_openimages_epoch28_withfilter.pt",
                "models",
            )
            self.shape = (3, 512, 512)
        elif name == "pixelart":
            self.model, self.scheduler = create_pixelart_model()
            checkpoint_path = load_file_from_url(
                "https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt",
                "models",
            )
            self.shape = (3, 256, 256)
        else:
            raise ValueError(f"Unknown model name {self.name}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.requires_grad_(False).eval()

        self.schedule_alphas = torch.nn.Parameter(
            torch.from_numpy(self.scheduler.alphas_cumprod).sqrt().float(),
            requires_grad=False,
        )

        self.schedule_sigmas = torch.nn.Parameter(
            (1 - torch.from_numpy(self.scheduler.alphas_cumprod)).sqrt().float(),
            requires_grad=False,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def schedule_indices(
        self, n_steps=500, from_index=999, to_index=0, rho=7.0
    ) -> lantern.Tensor:
        if from_index < to_index:
            raise ValueError("from_index must be greater than to_index")

        from_alpha, from_sigma = self.alphas(from_index), self.sigmas(from_index)
        to_alpha, to_sigma = self.alphas(to_index), self.sigmas(to_index)

        from_log_snr = torch.log(from_alpha**2 / from_sigma**2)
        to_log_snr = torch.log(to_alpha**2 / to_sigma**2)

        elucidated_from_sigma = (1 / from_log_snr.exp()).sqrt().clamp(max=150)
        elucidated_to_sigma = (1 / to_log_snr.exp()).sqrt().clamp(min=1e-3)

        ramp = torch.linspace(0, 1, n_steps + 1).to(self.device)
        min_inv_rho = elucidated_to_sigma ** (1 / rho)
        max_inv_rho = elucidated_from_sigma ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        target_log_snr = torch.log(torch.ones_like(sigmas) ** 2 / sigmas**2)

        schedule_log_snr = torch.log(
            self.schedule_alphas**2 / self.schedule_sigmas**2
        )

        assert target_log_snr.squeeze().ndim == 1
        assert schedule_log_snr.squeeze().ndim == 1
        schedule_indices = (
            (target_log_snr.squeeze()[:, None] - schedule_log_snr.squeeze()[None, :])
            .abs()
            .argmin(dim=1)
            .unique()
            .sort(descending=True)[0]
        )

        assert len(schedule_indices) >= n_steps * 0.9

        assert (schedule_indices[:-1] != schedule_indices[1:]).all()
        return torch.stack([schedule_indices[:-1], schedule_indices[1:]], dim=1)

    def random_diffused(self, shape) -> lantern.Tensor:
        n, c, h, w = shape
        if h % 8 != 0:
            raise ValueError("Height must be divisible by 32")
        if w % 8 != 0:
            raise ValueError("Width must be divisible by 32")
        return diffusion_space.decode(torch.randn(shape).to(self.device))

    def indices(self, indices) -> lantern.Tensor:
        if isinstance(indices, float) or isinstance(indices, int):
            indices = torch.as_tensor(indices)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1:
            raise ValueError("indices must be a scalar or a 1-dimensional tensor")
        return indices.long().to(self.device)

    def alphas(self, indices) -> lantern.Tensor:
        return self.schedule_alphas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    def sigmas(self, indices) -> lantern.Tensor:
        return self.schedule_sigmas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    @torch.cuda.amp.autocast()
    def predicted_noise(
        self,
        diffused_images,
        from_indices,
    ) -> lantern.Tensor:
        return self.model(
            diffusion_space.encode(diffused_images), self.indices(from_indices)
        )[:, :3].float()

    def predictions(self, diffused_images, indices) -> Predictions:
        indices = self.indices(indices)
        return Predictions(
            from_diffused_images=diffused_images,
            from_indices=indices,
            predicted_noise=self.predicted_noise(diffused_images, indices),
            schedule_alphas=self.schedule_alphas,
            schedule_sigmas=self.schedule_sigmas,
        )

    def forward(self, diffused_images, indices) -> Predictions:
        return self.predictions(diffused_images, indices)

    def diffuse_images(self, denoised_images, indices, noise=None) -> lantern.Tensor:
        indices = self.indices(indices)
        if noise is None:
            noise = torch.randn_like(denoised_images)
        alphas, sigmas = self.alphas(indices), self.sigmas(indices)
        return diffusion_space.decode(
            diffusion_space.encode(denoised_images) * alphas + noise * sigmas
        )


def test_guided_diffusion():
    from tqdm import tqdm
    from perceptor import utils

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    diffusion_model = GuidedDiffusion().to(device)
    diffused_images = diffusion_model.random_diffused((1, 3, 512, 512)).to(device)

    progress_bar = tqdm(
        diffusion_model.schedule_indices(to_index=0, n_steps=50, rho=3.0)
    )
    for from_indices, to_indices in progress_bar:
        step_predictions = diffusion_model.predictions(
            diffused_images,
            from_indices,
        )
        diffused_images = step_predictions.step(to_indices)

        utils.pil_image(step_predictions.denoised_images.clamp(0, 1)).save(
            "tests/guided_diffusion.png"
        )

        progress_bar.set_postfix(
            dict(
                from_indices=from_indices.item(),
                to_indices=to_indices.item(),
            )
        )

    predictions = diffusion_model.predictions(
        diffused_images,
        to_indices,
    )

    utils.pil_image(predictions.denoised_images.clamp(0, 1)).save(
        "tests/guided_diffusion.png"
    )


def test_guided_diffusion_init_image():
    import requests
    from PIL import Image
    import torch
    import torchvision.transforms.functional as TF
    from tqdm import tqdm
    from perceptor import utils

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    init_image = TF.to_tensor(
        Image.open(requests.get(image_url, stream=True).raw).resize((512, 512))
    )[None].cuda()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    from_index = 400

    diffusion_model = GuidedDiffusion().to(device)
    diffused_images = diffusion_model.diffuse_images(init_image, from_index)

    for from_indices, to_indices in tqdm(
        diffusion_model.schedule_indices(from_index=from_index, to_index=0, n_steps=50)
    ):
        for _ in range(4):
            predictions = diffusion_model.predictions(
                diffused_images,
                from_indices,
            )
            diffused_images = predictions.resample(to_indices)

        predictions = diffusion_model.predictions(
            diffused_images,
            from_indices,
        )
        diffused_images = predictions.step(to_indices)

    predictions = diffusion_model.predictions(
        diffused_images,
        to_indices,
    )

    utils.pil_image(predictions.denoised_images.clamp(0, 1)).save(
        "tests/guided_diffusion_init_image.png"
    )
