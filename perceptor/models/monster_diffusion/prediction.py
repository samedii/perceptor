from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor

from . import standardize, diffusion, settings


class Prediction(FunctionalBase):
    denoised_image: Tensor.dims("CHW").shape(*settings.INPUT_SHAPE)
    diffused_image: Tensor.dims("CHW").shape(*settings.INPUT_SHAPE)
    t: Tensor.dims("").float()

    def representation(self, example):
        denoised_image = self.denoised_image.detach().clamp(0, 1) * 255
        diffused_image = self.diffused_image.detach().clamp(0, 1) * 255
        horizontal_line = torch.full((3, 2, denoised_image.shape[-1]), 255)
        return np.uint8(
            torch.cat(
                [
                    torch.from_numpy(example.image).clamp(0, 255).permute(2, 0, 1),
                    horizontal_line,
                    diffused_image,
                    horizontal_line,
                    denoised_image,
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .numpy()
        )

    def pil_image(self, example):
        return Image.fromarray(self.representation(example))


class PredictionBatch(FunctionalBase):
    denoised_xs: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float()
    diffused_images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float()
    ts: Tensor.dims("N").float()

    def __len__(self):
        return len(self.denoised_images)

    def __getitem__(self, index):
        return Prediction(
            denoised_image=self.denoised_images[index],
            diffused_image=self.diffused_images[index],
            t=self.ts[index],
        )

    @property
    def device(self):
        return self.denoised_xs.device

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @staticmethod
    def sigmas(ts):
        if isinstance(ts, float):
            ts = torch.as_tensor(ts)
        if ts.ndim == 0:
            return torch.full((1,), ts).to(ts.device)
        return ts[:, None, None, None]

    @staticmethod
    def alphas(ts):
        return torch.ones_like(PredictionBatch.sigmas(ts))

    @property
    def from_sigmas(self):
        return self.sigmas(self.ts).to(self.device)

    @property
    def from_alphas(self):
        return self.alphas(self.ts).to(self.device)

    @property
    def from_xs(self):
        return self.diffused_xs

    @property
    def diffused_xs(self):
        return standardize.encode(self.diffused_images).to(self.device)

    @property
    def denoised_images(self):
        return standardize.decode(self.denoised_xs)

    @property
    def eps(self):
        return (self.diffused_xs - self.denoised_xs) / self.from_sigmas

    def step(self, to_ts):
        """
        Step the diffused image forward to `to_t`. Decreasing the amount of noise
        by moving closer to the predicted denoised image.
        """
        to_ts = to_ts.to(self.device)
        to_alphas, to_sigmas = self.alphas(to_ts), self.sigmas(to_ts)

        to_diffused_xs = self.denoised_xs * to_alphas + self.eps * to_sigmas
        return standardize.decode(to_diffused_xs)

        # to_diffused_xs = self.diffused_xs + self.eps * (to_sigmas - self.from_sigmas)
        # return standardize.decode(to_diffused_xs)

    def correction(self, previous_diffused_images, previous_ts, previous_eps):
        previous_diffused_xs = standardize.encode(
            previous_diffused_images.to(self.device)
        )
        corrected_diffused_xs = (
            previous_diffused_xs
            + (self.from_sigmas - self.sigmas(previous_ts))
            * (self.eps + previous_eps)
            / 2
        )
        return standardize.decode(corrected_diffused_xs)
