import math
import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.download_util import load_file_from_url

from .models import get_model
from .model_urls import model_urls
import perceptor.drawers.diffusion.utils as utils
from perceptor.drawers.interface import DrawingInterface


def prk_step(model, x, t_1, t_2, extra_args):
    def eps_model_fn(x, t, **extra_args):
        v = model(x, t, **extra_args)
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        eps = x * sigmas[:, None, None, None] + v * alphas[:, None, None, None]
        return eps

    ts = x.new_ones([x.shape[0]])
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(x, t_1 * ts, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(x_1, t_mid * ts, **extra_args)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(x_2, t_mid * ts, **extra_args)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(x_3, t_2 * ts, **extra_args)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def transfer(x, eps, t_1, t_2):
    alpha, sigma = utils.t_to_alpha_sigma(t_1)
    next_alpha, next_sigma = utils.t_to_alpha_sigma(t_2)
    pred = (x - eps * sigma) / alpha
    x = pred * next_alpha + eps * next_sigma
    return x, pred


class Diffusion(DrawingInterface):
    def __init__(self, init_images, n_steps=300, name="yfcc_2"):
        super().__init__()
        self.model = get_model(name)()
        checkpoint_path = load_file_from_url(model_urls[name], "models")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()
        self.model.requires_grad_(False)

        # TODO: add noise to image based on init_timestep
        # self.noisy_images = nn.Parameter(init_images, requires_grad=False)
        self.noisy_images = nn.Parameter(torch.randn_like(init_images))
        self.n_steps = n_steps
        self.timesteps = nn.Parameter(
            torch.linspace(1, 0, n_steps + 1)[:-1], requires_grad=False
        )
        self.steps = nn.Parameter(
            torch.cat(
                [
                    utils.get_spliced_ddpm_cosine_schedule(self.timesteps),
                    torch.zeros([1]),
                ]
            ),
            requires_grad=False,
        )

        # if args.init:
        #     steps = steps[steps < args.starting_timestep]
        #     alpha, sigma = utils.t_to_alpha_sigma(steps[0])
        #     x = init * alpha + x * sigma

        # cc12m_1_cfg takes clip encodings
        # self.clip_encodings

        self.eps_queue = list()

    def to(self, device):
        super().to(device)
        self.model.half()
        torch.cuda.amp.autocast()(self.model)
        return self

    @torch.no_grad()
    def synthesize(self, iteration):
        step1 = self.steps[iteration]
        step1_matrix = self.noisy_images.new_ones([self.noisy_images.shape[0]]) * step1
        v = self.model(self.noisy_images, step1_matrix)
        alphas, sigmas = utils.t_to_alpha_sigma(step1_matrix)
        pred = (
            self.noisy_images * alphas[:, None, None, None]
            - v * sigmas[:, None, None, None]
        )
        return (pred + 1) / 2

    def __iter__(self):
        for index in range(self.n_steps):
            yield self.synthesize_(index)

    @torch.cuda.amp.autocast()
    def synthesize_(self, index):
        if index == 0:
            # do not support guided synthesis during first 3 prk steps
            with torch.no_grad():
                for step1, step2 in zip(self.steps[:3], self.steps[1:4]):
                    self.noisy_images.data, eps_prime, _ = prk_step(
                        self.model, self.noisy_images, step1, step2, extra_args=dict()
                    )
                    self.eps_queue.append(eps_prime)

        prk_steps = 3
        index = index + prk_steps

        # def plms_sample(model, x, steps)
        self.step1 = self.steps[index]
        self.step2 = self.steps[index + 1]
        # plms_step(model_fn, x, old_eps, steps[i], steps[i + 1])
        # def plms_step(model, x, old_eps, t_1, t_2)
        step1_matrix = (
            self.noisy_images.new_ones([self.noisy_images.shape[0]]) * self.step1
        )
        # eps = eps_model_fn(self.noisy_images, step1_matrix)
        # def eps_model_fn(x, t)
        # v = model(x, t)
        # def cond_model_fn(x, t)
        self.v = self.model(self.noisy_images, step1_matrix)
        self.alphas, self.sigmas = utils.t_to_alpha_sigma(step1_matrix)
        pred = (
            self.noisy_images * self.alphas[:, None, None, None]
            - self.v * self.sigmas[:, None, None, None]
        )
        # cond_grad = cond_fn(x, t, pred, **extra_args).detach()
        # def cond_fn(x, t, pred, clip_embed)
        return (pred + 1) / 2

        # this is what we want outside the drawer
        # backprop losses

    @torch.no_grad()
    def step_(self):
        # cond_model_fn after
        cond_grad = -self.noisy_images.grad
        v = self.v - cond_grad * (
            self.sigmas[:, None, None, None] / self.alphas[:, None, None, None]
        )

        # eps_model_fn after
        # alphas, sigmas = utils.t_to_alpha_sigma(step1_matrix)
        eps = (
            self.noisy_images * self.sigmas[:, None, None, None]
            + v * self.alphas[:, None, None, None]
        )

        # plms_step after
        eps = (
            55 * eps
            - 59 * self.eps_queue[-1]
            + 37 * self.eps_queue[-2]
            - 9 * self.eps_queue[-3]
        ) / 24
        self.noisy_images.data, pred = transfer(
            self.noisy_images, eps, self.step1, self.step2
        )

        # plms_sample after
        self.eps_queue.pop(0)
        self.eps_queue.append(eps)
        return (pred + 1) / 2

    def encode(self, images, mode="bilinear"):
        raise NotImplementedError

    def replace_(self, noisy_images):
        self.noisy_images.data.copy_(noisy_images.data)
        return self
