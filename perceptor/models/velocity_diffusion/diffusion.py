from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.download_util import load_file_from_url

from .models import get_model
from .model_urls import model_urls
import perceptor.drawers.diffusion.utils as utils
from perceptor.drawers.interface import DrawingInterface


def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, t, **extra_args):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            v = model(x, t, **extra_args)
            alphas, sigmas = utils.t_to_alpha_sigma(t)
            pred = x * alphas[:, None, None, None] - v * sigmas[:, None, None, None]
            cond_grad = cond_fn(x, t, pred, **extra_args).detach()
            v = v.detach() - cond_grad * (
                sigmas[:, None, None, None] / alphas[:, None, None, None]
            )
        return v

    return cond_model_fn


def make_eps_model_fn(model):
    def eps_model_fn(x, t, **extra_args):
        v = model(x, t, **extra_args)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        eps = x * sigmas[:, None, None, None] + v * alphas[:, None, None, None]
        return eps

    return eps_model_fn


def transfer(x, eps, t_1, t_2):
    alphas, sigmas = utils.t_to_alpha_sigma(t_1)
    next_alphas, next_sigmas = utils.t_to_alpha_sigma(t_2)
    pred = (x - eps * sigmas[:, None, None, None]) / alphas[:, None, None, None]
    x = pred * next_alphas[:, None, None, None] + eps * next_sigmas[:, None, None, None]
    return x, pred


def prk_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(x_1, t_mid, **extra_args)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(x_2, t_mid, **extra_args)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(x_3, t_2, **extra_args)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps, pred


class Diffusion(DrawingInterface):
    def __init__(self, shape, name="yfcc_2", n_steps=50):
        super().__init__()
        self.n_steps = n_steps

        self.model = get_model(name)()
        checkpoint_path = load_file_from_url(model_urls[name], "models")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint {checkpoint_path}")
        self.model.eval()
        self.model.requires_grad_(False)

        # self.model.clip_model
        self.latent = nn.Parameter(torch.randn(shape))
        self.noise = 1.0

        # if args.init:
        #     steps = steps[steps < args.starting_timestep]
        #     alpha, sigma = utils.t_to_alpha_sigma(steps[0])
        #     x = init * alpha + x * sigma

    def to(self, device):
        super().to(device)
        if device == torch.device("cuda"):
            self.model.half()
        else:
            print("Warning: model is not converted to half precision")
        return self

    @torch.no_grad()
    def step_(self, velocity):
        # take a step in direction x, then reverse back (add noise)
        pass

    @torch.cuda.amp.autocast()
    def synthesize(self, noise=None):
        if noise is None or noise == self.noise:
            ones = torch.ones(self.latent.shape[:1]).to(self.latent)
            with torch.no_grad():
                if hasattr(self.model, "clip_model"):
                    velocity = self.model(
                        self.latent, self.noise * ones, clip_embed=self.clip_encodings
                    )
                else:
                    velocity = self.model(self.latent, self.noise * ones)
            alphas, sigmas = utils.t_to_alpha_sigma(self.noise * ones)
            pred = (
                self.latent * alphas[:, None, None, None]
                - velocity * sigmas[:, None, None, None]
            )
            return (pred + 1) / 2
        else:
            if noise >= self.noise:
                raise ValueError(
                    f"Increasing noise is not supported. Tried to change noise {self.noise} -> {noise}"
                )

            if hasattr(self.model, "clip_model"):
                extra_args = dict(clip_embed=self.clip_encodings)
            else:
                extra_args = {}

            model_fn = torch.cuda.amp.autocast()(self.model)

            t = torch.linspace(1, 0, self.n_steps + 1)[:-1]
            steps = utils.get_spliced_ddpm_cosine_schedule(t)
            steps = torch.cat([steps, steps.new_zeros([1])])
            steps = steps[(steps <= self.noise) & (steps >= noise)]

            latent = self.latent
            ones = torch.ones(latent.shape[:1]).to(latent)
            eps_queue = []
            for from_step, to_step in zip(steps, steps[1:]):
                if len(eps_queue) < 3:
                    latent, eps, pred = prk_step(
                        model_fn, latent, from_step * ones, to_step * ones, extra_args
                    )
                else:
                    latent, eps, pred = plms_step(
                        model_fn,
                        latent,
                        eps_queue,
                        from_step * ones,
                        to_step * ones,
                        extra_args,
                    )
                    eps_queue.pop(0)
                eps_queue.append(eps)
            return (latent + 1) / 2

    @torch.no_grad()
    def inverse_(self, noise):
        if noise >= self.noise:
            raise ValueError(
                f"Increasing noise is not supported. Tried to change noise {self.noise} -> {noise}"
            )

        if hasattr(self.model, "clip_model"):
            extra_args = dict(clip_embed=self.clip_encodings)
        else:
            extra_args = {}

        model_fn = torch.cuda.amp.autocast()(self.model)

        t = torch.linspace(1, 0, self.n_steps + 1)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        steps = torch.cat([steps, steps.new_zeros([1])])
        steps = steps[(steps <= self.noise) & (steps >= noise)]

        ones = torch.ones(self.latent.shape[:1]).to(self.latent)
        eps_queue = []
        for from_step, to_step in zip(steps, steps[1:]):
            if len(eps_queue) < 3:
                self.latent.data, eps, pred = prk_step(
                    model_fn, self.latent, from_step * ones, to_step * ones, extra_args
                )
            else:
                self.latent.data, eps, pred = plms_step(
                    model_fn,
                    self.latent,
                    eps_queue,
                    from_step * ones,
                    to_step * ones,
                    extra_args,
                )
                eps_queue.pop(0)
            eps_queue.append(eps)
            self.noise = to_step.item()
            yield torch.clamp((pred + 1) / 2, 0, 1)

    def encode(self, images, noise=None):
        if noise is None:
            noise = self.noise
        ones = torch.ones(self.latent.shape[:1]).to(self.latent)
        alpha, sigma = utils.t_to_alpha_sigma(noise * ones)
        return (images * 2 - 1) * alpha + torch.randn_like(images) * sigma

    def replace_(self, latent):
        self.latent.data.copy_(latent.data)
        return self
