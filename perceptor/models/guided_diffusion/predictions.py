from __future__ import annotations
import torch
import lantern

from perceptor.transforms.clamp_with_grad import clamp_with_grad
from . import diffusion_space


class Predictions(lantern.FunctionalBase):
    from_diffused_images: lantern.Tensor.dims("NCHW")
    from_indices: lantern.Tensor.dims("N")
    predicted_noise: lantern.Tensor.dims("NCHW")
    schedule_alphas: lantern.Tensor.dims("N")
    schedule_sigmas: lantern.Tensor.dims("N")

    @property
    def device(self):
        return self.predicted_noise.device

    def indices(self, indices) -> lantern.Tensor:
        if isinstance(indices, float) or isinstance(indices, int):
            indices = torch.as_tensor(indices)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1:
            raise ValueError("indices must be a scalar or a 1D tensor")
        return indices.long().to(self.device)

    def alphas(self, indices) -> lantern.Tensor:
        return self.schedule_alphas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    def sigmas(self, indices) -> lantern.Tensor:
        return self.schedule_sigmas[self.indices(indices)][:, None, None, None].to(
            self.device
        )

    @property
    def from_alphas(self) -> lantern.Tensor:
        return self.alphas(self.from_indices)

    @property
    def from_sigmas(self) -> lantern.Tensor:
        return self.sigmas(self.from_indices)

    @property
    def from_diffused_xs(self) -> lantern.Tensor:
        return diffusion_space.encode(self.from_diffused_images)

    @property
    def denoised_xs(self) -> lantern.Tensor:
        return (
            self.from_diffused_xs - self.from_sigmas * self.predicted_noise
        ) / self.from_alphas.clamp(min=1e-7)

    @property
    def denoised_images(self) -> lantern.Tensor:
        return diffusion_space.decode(self.denoised_xs)

    def step(self, to_indices, eta=0.0) -> lantern.Tensor:
        """
        Reduce noise level to `to_indices`

        Args:
            to_indices: Union[Tensor, Tensor.shape("N"), float]
            eta: float

        Returns:
            diffused_images: torch.Tensor.shape("NCHW")
        """
        to_alphas, to_sigmas = self.alphas(to_indices), self.sigmas(to_indices)

        if eta > 0.0:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = (
                eta
                * (to_sigmas**2 / self.from_sigmas**2).sqrt()
                * (1 - self.from_alphas**2 / to_alphas**2).sqrt()
            )
            adjusted_sigma = (to_sigmas**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            to_diffused_xs = (
                self.denoised_xs * to_alphas + self.predicted_noise * adjusted_sigma
            )

            # Add the correct amount of fresh noise
            noise = torch.randn_like(to_diffused_xs)
            to_diffused_xs += noise * ddim_sigma
        else:
            to_diffused_xs = (
                self.denoised_xs * to_alphas + self.predicted_noise * to_sigmas
            )

        return diffusion_space.decode(to_diffused_xs)
        # TODO: do not need to calculate denoised images? this could introduce errors?

    def correction(self, previous: Predictions) -> Predictions:
        return previous.forced_denoised(
            (self.denoised_images + previous.denoised_images) / 2
        )

    def reverse_step(self, to_indices) -> lantern.Tensor:
        if (torch.as_tensor(self.from_indices) > torch.as_tensor(to_indices)).any():
            raise ValueError("from_indices must be less than to_indices")

        to_alphas, to_sigmas = self.alphas(to_indices), self.sigmas(to_indices)
        return diffusion_space.decode(
            self.denoised_xs * to_alphas + self.predicted_noise * to_sigmas
        )
        # TODO: do not need to calculate denoised images? this could introduce errors?

    def resample(self, resample_indices) -> lantern.Tensor:
        """
        Harmonizing resampling from https://github.com/andreas128/RePaint
        """
        return diffusion_space.decode(
            self.denoised_xs * self.from_alphas
            + self.resample_noise(resample_indices) * self.from_sigmas
        )

    def resample_noise(self, resample_indices) -> lantern.Tensor:
        if (
            torch.as_tensor(self.from_indices) < torch.as_tensor(resample_indices)
        ).any():
            raise ValueError("from_indices must be greater than resample_indices")
        resampled_noise_sigma = (
            self.sigmas(resample_indices) * self.predicted_noise
            + (
                self.from_sigmas**2 - self.sigmas(resample_indices) ** 2
            ).sqrt() * torch.randn_like(self.predicted_noise)
        )  # fmt: skip
        return resampled_noise_sigma / self.from_sigmas

    def noisy_reverse_step(self, to_indices) -> lantern.Tensor:
        to_alphas, to_sigmas = self.alphas(to_indices), self.sigmas(to_indices)

        noise_sigma = self.from_sigmas * self.predicted_noise + (
            to_sigmas**2 - self.from_sigmas**2
        ).sqrt() * torch.randn_like(self.predicted_noise)

        return diffusion_space.decode(self.denoised_xs * to_alphas + noise_sigma)

    def guided(self, guiding, guidance_scale=0.5, clamp_value=1e-6) -> Predictions:
        return self.replace(
            predicted_noise=self.predicted_noise
            + guidance_scale
            * self.from_sigmas
            * guiding.clamp(-clamp_value, clamp_value)
            / clamp_value
        )

    def dynamic_threshold(self, quantile=0.95) -> Predictions:
        """
        Thresholding heuristic from imagen paper
        """
        dynamic_threshold = torch.quantile(
            self.denoised_xs.flatten(start_dim=1).abs(), quantile, dim=1
        ).clamp(min=1.0)
        denoised_xs = (
            clamp_with_grad(
                self.denoised_xs,
                -dynamic_threshold,
                dynamic_threshold,
            )
            # / dynamic_threshold
            # imagen's dynamic thresholding divides by threshold but this makes the images gray
        )
        return self.forced_denoised_images(diffusion_space.decode(denoised_xs))

    def forced_denoised_images(self, denoised_images) -> Predictions:
        denoised_xs = diffusion_space.encode(denoised_images)
        predicted_noise = (
            self.from_diffused_xs - denoised_xs * self.from_alphas
        ) / self.from_sigmas.clamp(min=1e-7)
        return self.replace(predicted_noise=predicted_noise)

    def forced_predicted_noise(self, predicted_noise) -> Predictions:
        return self.replace(predicted_noise=predicted_noise)

    def wasserstein_distance(self) -> lantern.Tensor:
        sorted_noise = self.predicted_noise.flatten(start_dim=1).sort(dim=1)[0]
        n = sorted_noise.shape[1]
        margin = 0.5 / n
        points = torch.linspace(margin, 1 - margin, sorted_noise.shape[1])
        expected_noise = torch.distributions.Normal(0, 1).icdf(points)
        return (sorted_noise - expected_noise[None].to(sorted_noise)).abs().mean()

    def wasserstein_square_distance(self) -> lantern.Tensor:
        sorted_noise = self.predicted_noise.flatten(start_dim=1).sort(dim=1)[0]
        n = sorted_noise.shape[1]
        margin = 0.5 / n
        points = torch.linspace(margin, 1 - margin, sorted_noise.shape[1])
        expected_noise = torch.distributions.Normal(0, 1).icdf(points)
        return (sorted_noise - expected_noise[None].to(sorted_noise)).square().mean()
