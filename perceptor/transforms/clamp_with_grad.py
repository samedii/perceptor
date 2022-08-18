import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .interface import TransformInterface


class ClampWithGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min=0, max=1):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


def clamp_with_grad(tensor, min=0.0, max=1.0):
    return ClampWithGradFunction.apply(tensor, min, max)


class ClampWithGrad(TransformInterface):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min = min
        self.max = max

    def encode(self, tensor):
        return clamp_with_grad(tensor, self.min, self.max)

    def decode(self, tensor):
        return tensor
