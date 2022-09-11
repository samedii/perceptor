import torch
import lantern

from . import diffusion_space, utils
from perceptor.transforms.clamp_with_grad import clamp_with_grad


class Predictions(lantern.FunctionalBase):
    from_diffused_images: lantern.Tensor.dims("NCHW")
    from_ts: lantern.Tensor.dims("N")
    velocities: lantern.Tensor.dims("NCHW")

    @property
    def device(self):
        return self.velocities.device

    def alphas(self, ts):
        if isinstance(ts, float):
            ts = torch.tensor(ts)
        if ts.ndim == 0:
            ts = ts[None]
        if ts.ndim != 1:
            raise ValueError("ts must be a scalar or a 1D tensor")
        alphas, _ = utils.t_to_alpha_sigma(ts)
        return alphas[:, None, None, None].to(self.device)

    def sigmas(self, ts):
        if isinstance(ts, float):
            ts = torch.tensor(ts)
        if ts.ndim == 0:
            ts = ts[None]
        if ts.ndim != 1:
            raise ValueError("ts must be a scalar or a 1D tensor")
        _, sigmas = utils.t_to_alpha_sigma(ts)
        return sigmas[:, None, None, None].to(self.device)

    @property
    def from_alphas(self):
        return self.alphas(self.from_ts)

    @property
    def from_sigmas(self):
        return self.sigmas(self.from_ts)

    @property
    def from_diffused_xs(self):
        return diffusion_space.encode(self.from_diffused_images)

    @property
    def denoised_xs(self):
        return (
            self.from_diffused_xs * self.from_alphas
            - self.velocities * self.from_sigmas
        )

    @property
    def predicted_noise(self):
        return (
            self.from_diffused_xs * self.from_sigmas
            + self.velocities * self.from_alphas
        )

    @property
    def denoised_images(self):
        return diffusion_space.decode(self.denoised_xs)

    def step(self, to_ts, eta=0.0):
        """
        Reduce noise level to `to_ts`

        Args:
            to_ts: Union[Tensor, Tensor.shape("N"), float]
            eta: float

        Returns:
            diffused_images: torch.Tensor.shape("NCHW")
        """
        to_alphas, to_sigmas = self.alphas(to_ts), self.sigmas(to_ts)

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

    def correction(self, previous: "Predictions"):
        return previous.forced_denoised(
            (self.denoised_images + previous.denoised_images) / 2
        )

    def reverse_step(self, to_ts):
        if (torch.as_tensor(self.from_ts) > torch.as_tensor(to_ts)).any():
            raise ValueError("from_ts must be less than to_ts")

        to_alphas, to_sigmas = self.alphas(to_ts), self.sigmas(to_ts)
        return self.denoised_xs * to_alphas + self.predicted_noise * to_sigmas

    def resample_noise(self, resample_ts):
        if (torch.as_tensor(self.from_ts) < torch.as_tensor(resample_ts)).any():
            raise ValueError("from_ts must be greater than resample_ts")
        resampled_noise_sigma = (
            self.sigmas(resample_ts) * self.predicted_noise
            + (
                self.from_sigmas**2 - self.sigmas(resample_ts) ** 2
            ).sqrt() * torch.randn_like(self.predicted_noise)
        )  # fmt: skip
        return resampled_noise_sigma / self.from_sigmas

    def resample(self, resample_ts):
        """
        Harmonizing resampling from https://github.com/andreas128/RePaint
        """
        return diffusion_space.decode(
            self.denoised_xs * self.from_alphas
            + self.resample_noise(resample_ts) * self.from_sigmas
        )

    def noisy_reverse_step(self, to_ts):
        to_alphas, to_sigmas = self.alphas(to_ts), self.sigmas(to_ts)

        noise_sigma = self.from_sigmas * self.predicted_noise + (
            to_sigmas**2 - self.from_sigmas**2
        ).sqrt() * torch.randn_like(self.predicted_noise)

        return diffusion_space.decode(self.denoised_xs * to_alphas + noise_sigma)

    def guided(self, guiding, guidance_scale=0.5, clamp_value=1e-6) -> "Predictions":
        return self.replace(
            velocities=self.velocities
            + guidance_scale
            * self.from_sigmas
            * guiding.clamp(-clamp_value, clamp_value)
            / clamp_value
        )

    def dynamic_threshold(self, quantile=0.95) -> "Predictions":
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
        return self.forced_denoised(diffusion_space.decode(denoised_xs))

    def static_threshold(self):
        return self.forced_denoised(clamp_with_grad(self.denoised_images, 0, 1))

    def forced_denoised(self, denoised_images) -> "Predictions":
        denoised_xs = diffusion_space.encode(denoised_images)
        if (self.from_sigmas >= 1e-3).all():
            predicted_noise = (
                self.from_diffused_xs - denoised_xs * self.from_alphas
            ) / self.from_sigmas
        else:
            predicted_noise = self.predicted_noise
        return self.replace(
            velocities=self.from_alphas * predicted_noise
            - self.from_sigmas * denoised_xs
        )

    def forced_predicted_noise(self, predicted_noise) -> "Predictions":
        if (self.from_alphas >= 1e-3).all():
            denoised_xs = (
                self.from_diffused_xs - predicted_noise * self.from_sigmas
            ) / self.from_alphas
        else:
            denoised_xs = self.denoised_xs
        return self.replace(
            velocities=self.from_alphas * predicted_noise
            - self.from_sigmas * denoised_xs
        )

    def wasserstein_distance(self):
        sorted_noise = self.predicted_noise.flatten(start_dim=1).sort(dim=1)[0]
        n = sorted_noise.shape[1]
        margin = 0.5 / n
        points = torch.linspace(margin, 1 - margin, sorted_noise.shape[1])
        expected_noise = torch.distributions.Normal(0, 1).icdf(points)
        return (sorted_noise - expected_noise[None].to(sorted_noise)).abs().mean()

    def wasserstein_square_distance(self):
        sorted_noise = self.predicted_noise.flatten(start_dim=1).sort(dim=1)[0]
        n = sorted_noise.shape[1]
        margin = 0.5 / n
        points = torch.linspace(margin, 1 - margin, sorted_noise.shape[1])
        expected_noise = torch.distributions.Normal(0, 1).icdf(points)
        return (sorted_noise - expected_noise[None].to(sorted_noise)).square().mean()
