from typing import Optional
from tqdm import tqdm
from scipy import integrate
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import module_device, Tensor
from basicsr.utils.download_util import load_file_from_url

from . import settings
from . import standardize
from . import base
from . import diffusion
from .prediction import PredictionBatch

ALL_CHECKPOINT_URL = "https://s3.eu-central-1.wasabisys.com/nextml-model-data/monster-diffusion/6b70ff1e6c7f4c00ad8cb59879f7d88d.pt"
TINY_HERO_CHECKPOINT_URL = "https://s3.eu-central-1.wasabisys.com/nextml-model-data/monster-diffusion/f47af8975b744d4bae2b905bac223003.pt"


class MonsterDiffusion(nn.Module):
    def __init__(self, name="all"):
        super().__init__()
        self.network = base.Model(
            mapping_cond_dim=9,
        )
        if name == "all":
            checkpoint_path = load_file_from_url(ALL_CHECKPOINT_URL, "models")
        elif name == "tiny-hero":
            checkpoint_path = load_file_from_url(TINY_HERO_CHECKPOINT_URL, "models")
        else:
            raise ValueError(f"Unknown model name {name}")
        self.load_state_dict(torch.load(checkpoint_path))
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return module_device(self)

    @staticmethod
    def training_ts(size):
        random_ts = (diffusion.P_mean + torch.randn(size) * diffusion.P_std).exp()
        return random_ts

    def _schedule_ts(self, n_steps):
        ramp = torch.linspace(0, 1, n_steps).to(self.device)
        min_inv_rho = diffusion.sigma_min ** (1 / diffusion.rho)
        max_inv_rho = diffusion.sigma_max ** (1 / diffusion.rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** diffusion.rho

    def schedule_ts(self, n_steps):
        schedule_ts = self._schedule_ts(n_steps)
        return zip(schedule_ts[:-1], schedule_ts[1:])

    @staticmethod
    def sigmas(ts):
        return PredictionBatch.sigmas(ts)

    @staticmethod
    def alphas(ts):
        return PredictionBatch.alphas(ts)

    @staticmethod
    def random_noise(size):
        return standardize.decode(
            torch.randn(size, *settings.INPUT_SHAPE) * diffusion.sigma_max
        )

    @staticmethod
    def diffuse(
        images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
        ts,
        noise=None,
    ):
        x0 = standardize.encode(images)
        if isinstance(ts, float) or ts.ndim == 0:
            ts = torch.full((x0.shape[0],), ts).to(x0.device)

        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)

        assert x0.shape == noise.shape

        return standardize.decode(x0 + noise * MonsterDiffusion.sigmas(ts))

    def c_skip(self, ts):
        return diffusion.sigma_data**2 / (
            diffusion.sigma_data**2 + self.sigmas(ts) ** 2
        )

    def c_out(self, ts):
        return (
            self.sigmas(ts)
            * diffusion.sigma_data
            / torch.sqrt(diffusion.sigma_data**2 + self.sigmas(ts) ** 2)
        )

    def c_in(self, ts):
        return 1 / torch.sqrt(diffusion.sigma_data**2 + self.sigmas(ts) ** 2)

    def c_noise(self, ts):
        return 1 / 4 * self.sigmas(ts).log().view(-1)

    def denoised_(
        self,
        diffused_images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Optional[Tensor.dims("NK")] = None,
    ) -> Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE):
        """
        Parameterization from https://arxiv.org/pdf/2206.00364.pdf
        """
        diffused_xs = standardize.encode(diffused_images.to(self.device))
        ts = ts.to(self.device)
        if nonleaky_augmentations is None:
            nonleaky_augmentations = torch.zeros(
                len(diffused_images), settings.N_AUGMENTATIONS
            )
        nonleaky_augmentations = nonleaky_augmentations.to(self.device)

        output = self.network(
            self.c_in(ts) * diffused_xs,
            self.c_noise(self.sigmas(ts).flatten()),
            mapping_cond=nonleaky_augmentations,
        )
        return self.c_skip(ts) * diffused_xs + self.c_out(ts) * output

    def forward(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Optional[Tensor.dims("NK")] = None,
    ):
        denoised_xs = self.denoised_(
            diffused_images,
            ts,
            nonleaky_augmentations,
        )
        return PredictionBatch(
            denoised_xs=denoised_xs,
            diffused_images=diffused_images,
            ts=torch.as_tensor(ts).flatten().to(self.device),
        )

    def predictions_(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Optional[Tensor.dims("NK")] = None,
    ):
        return self.forward(
            diffused_images,
            ts,
            nonleaky_augmentations,
        )

    def predictions(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Optional[Tensor.dims("NK")] = None,
    ):
        if self.training:
            raise Exception(
                "Cannot run predictions method while in training mode. Use predictions_"
            )
        return self.predictions_(
            diffused_images,
            ts,
            nonleaky_augmentations,
        )

    @staticmethod
    def gamma(ts, n_steps):
        return torch.where(
            (ts >= diffusion.S_tmin) & (ts <= diffusion.S_tmax),
            torch.minimum(
                torch.tensor(diffusion.S_churn / n_steps),
                torch.tensor(2).sqrt() - 1,
            ).to(ts),
            torch.zeros_like(ts),
        )

    @staticmethod
    def reversed_ts(ts, n_steps):
        return ts + MonsterDiffusion.gamma(ts, n_steps) * ts

    def inject_noise(self, diffused_images, ts, reversed_ts):
        diffused_xs = standardize.encode(diffused_images).to(self.device)

        reversed_diffused_xs = (
            diffused_xs
            + (self.sigmas(reversed_ts).square() - self.sigmas(ts).square()).sqrt()
            * torch.randn_like(diffused_xs)
            * diffusion.S_noise
        )
        return standardize.decode(reversed_diffused_xs)

    def sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        diffused_images=None,
    ):
        return self.elucidated_sample(
            size,
            n_evaluations,
            progress,
            diffused_images,
        )

    def elucidated_sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        diffused_images=None,
    ):
        """
        Elucidated stochastic sampling from https://arxiv.org/pdf/2206.00364.pdf
        """
        if self.training:
            raise Exception("Cannot run sample method while in training mode.")
        if diffused_images is None:
            diffused_images = self.random_noise(size).to(self.device)
        nonleaky_augmentations = torch.zeros(
            (size, settings.N_AUGMENTATIONS), dtype=torch.float32, device=self.device
        )

        n_steps = n_evaluations // 2
        i = 0
        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for from_ts, to_ts in self.schedule_ts(n_steps):
            reversed_ts = self.reversed_ts(from_ts, n_steps).clamp(
                max=diffusion.sigma_max
            )
            reversed_diffused_images = self.inject_noise(
                diffused_images, from_ts, reversed_ts
            )
            i += 1

            predictions = self.predictions(
                reversed_diffused_images,
                reversed_ts,
                nonleaky_augmentations,
            )
            reversed_eps = predictions.eps
            diffused_images = predictions.step(to_ts)

            predictions = self.predictions(
                diffused_images,
                to_ts,
                nonleaky_augmentations,
            )
            diffused_images = predictions.correction(
                reversed_diffused_images, reversed_ts, reversed_eps
            )
            progress.update()
            yield predictions.denoised_images.clamp(0, 1)

        reversed_ts = self.reversed_ts(to_ts, n_steps)
        diffused_images = self.inject_noise(diffused_images, to_ts, reversed_ts)

        predictions = self.predictions(
            diffused_images,
            reversed_ts,
            nonleaky_augmentations,
        )
        progress.close()
        yield predictions.denoised_images.clamp(0, 1)

    @staticmethod
    def linear_multistep_coeff(order, sigmas, from_index, to_index):
        if order - 1 > from_index:
            raise ValueError(f"Order {order} too high for step {from_index}")

        def fn(tau):
            prod = 1.0
            for k in range(order):
                if to_index == k:
                    continue
                prod *= (tau - sigmas[from_index - k]) / (
                    sigmas[from_index - to_index] - sigmas[from_index - k]
                )
            return prod

        return integrate.quad(
            fn, sigmas[from_index], sigmas[from_index + 1], epsrel=1e-4
        )[0]

    def linear_multistep_sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        diffused_images=None,
        order=4,
    ):
        """
        Katherine Crowson's linear multistep method from https://github.com/crowsonkb/k-diffusion/blob/4fdb34081f7a09f16c33d3344a042e5bea8e69ee/k_diffusion/sampling.py
        """
        if self.training:
            raise Exception("Cannot run sample method while in training mode.")
        if diffused_images is None:
            diffused_images = self.random_noise(size)
        nonleaky_augmentations = torch.zeros(
            (size, settings.N_AUGMENTATIONS), dtype=torch.float32, device=self.device
        )
        diffused_images = diffused_images.to(self.device)

        n_steps = n_evaluations
        schedule_ts = self._schedule_ts(n_steps)

        epses = list()
        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for from_index, (from_ts, to_ts) in enumerate(self.schedule_ts(n_steps)):

            predictions = self.predictions(
                diffused_images,
                from_ts,
                nonleaky_augmentations,
            )
            epses.append(predictions.eps)
            if len(epses) > order:
                epses.pop(0)

            current_order = len(epses)
            coeffs = [
                self.linear_multistep_coeff(
                    current_order,
                    self.sigmas(schedule_ts).cpu().flatten(),
                    from_index,
                    to_index,
                )
                for to_index in range(current_order)
            ]

            diffused_xs = standardize.encode(diffused_images)
            diffused_xs = diffused_xs + sum(
                coeff * eps for coeff, eps in zip(coeffs, reversed(epses))
            )
            diffused_images = standardize.decode(diffused_xs)

            progress.update()
            yield predictions.denoised_images.clamp(0, 1)

        predictions = self.predictions(
            diffused_images,
            to_ts,
            nonleaky_augmentations,
        )
        progress.close()
        yield predictions.denoised_images.clamp(0, 1)


def test_monster_diffusion():
    from perceptor import utils

    model = MonsterDiffusion().cuda()
    for images in model.sample(size=1, n_evaluations=50):
        pass
    utils.pil_image(images).save("tests/monster_diffusion.png")


def test_monster_diffusion_lms():
    from perceptor import utils

    model = MonsterDiffusion().cuda()
    for images in model.linear_multistep_sample(size=1, n_evaluations=50):
        pass
    utils.pil_image(images).save("tests/monster_diffusion_lms.png")
