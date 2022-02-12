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


class DiffusionOriginal(DrawingInterface):
    def __init__(self, init_images, name="yfcc_2", clip_encodings=None):
        super().__init__()
        self.model = get_model(name)()
        checkpoint_path = load_file_from_url(model_urls[name], "models")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint {checkpoint_path}")
        self.model.eval()
        self.model.requires_grad_(False)

        # self.clip_model = self.model.clip_model

        if clip_encodings is None:
            self.clip_encodings = None
        else:
            self.clip_encodings = nn.Parameter(clip_encodings, requires_grad=False)

        # TODO: add noise to image based on init_timestep
        self.latent = nn.Parameter(torch.randn_like(init_images))

        # if args.init:
        #     steps = steps[steps < args.starting_timestep]
        #     alpha, sigma = utils.t_to_alpha_sigma(steps[0])
        #     x = init * alpha + x * sigma

    # def encode_texts(self, text):
    #     return self.model.clip_model

    def to(self, device):
        super().to(device)
        self.model.half()
        return self

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def synthesize(self):
        step1_matrix = torch.full(self.latent.shape[:1], self.from_step).to(self.latent)
        if hasattr(self.model, "clip_model"):
            v = self.model(self.latent, step1_matrix, clip_embed=self.clip_encodings)
        else:
            v = self.model(self.latent, step1_matrix)
        alphas, sigmas = utils.t_to_alpha_sigma(step1_matrix)
        pred = (
            self.latent * alphas[:, None, None, None] - v * sigmas[:, None, None, None]
        )
        return ((pred + 1) / 2).clamp(0, 1)

    @torch.no_grad()
    def inverse_(self, guide, n_steps):
        yielded_pred = None

        def cond_fn(x, t, pred, clip_embed=None):
            nonlocal yielded_pred
            yielded_pred = pred.detach().clone()
            loss = guide((pred + 1) / 2)
            # cond_grad = -torch.autograd.grad(loss, x)[0]
            cond_grad = -x.grad
            # x.grad.detach_().zero_()
            return cond_grad

        if hasattr(self.model, "clip_model"):
            extra_args = dict(clip_embed=self.clip_encodings)
        else:
            extra_args = {}

        cond_model_fn = make_cond_model_fn(self.model, cond_fn)
        model_fn = torch.cuda.amp.autocast()(cond_model_fn)

        t = torch.linspace(1, 0, n_steps + 1)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        steps = torch.cat([steps, steps.new_zeros([1])])
        ones = torch.ones(self.latent.shape[:1]).to(self.latent)

        # if args.init:
        #     steps = steps[steps < args.starting_timestep]
        #     alpha, sigma = utils.t_to_alpha_sigma(steps[0])
        #     self.latent = init * alpha + self.latent * sigma

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
            self.from_step = to_step
            yield torch.clamp((yielded_pred + 1) / 2, 0, 1)

    def encode(self, images):
        raise NotImplementedError

    def replace_(self, latent):
        self.latent.data.copy_(latent.data)
        return self
