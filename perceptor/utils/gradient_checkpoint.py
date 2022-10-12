from lantern import FunctionalBase, Tensor
import torch


class GradientCheckpoint(FunctionalBase):
    original: Tensor
    detached: Tensor

    def __init__(self, tensor):
        super().__init__(original=tensor, detached=tensor.detach().requires_grad_())

    def zero_grad_(self):
        self.detached.grad.zero_()
        return self

    def backward(self, loss):
        loss.backward()
        gradients = self.detached.grad.clone()
        self.zero_grad_()
        return gradients

    def continue_backward(self, gradients=None, retain_graph=False):
        if self.detached.grad is None:
            raise ValueError("Gradient is not defined")

        if gradients is None:
            return self.original.backward(self.detached.grad, retain_graph=retain_graph)
        else:
            return self.original.backward(gradients, retain_graph=retain_graph)

    def tensor(self):
        return self.detached

    @staticmethod
    def nonzero_mean(gradients, dim=0):
        if isinstance(gradients, list):
            gradients = torch.stack(gradients)
        return gradients.sum(dim).div(gradients.ne(0).sum(dim).add(1e-6))

    @staticmethod
    def nonzero_scale(tensor, dim=None):
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)
        shape = tensor.shape
        if dim is None:
            tensor = tensor.flatten()
            dim = 0

        mask = tensor.ne(0)
        mean_square = tensor.square().sum(dim) / mask.sum(dim).add(1e-6)
        mean = tensor.sum(dim) / mask.sum(dim).add(1e-6)
        std = (mean_square - mean.square()).sqrt().add(1e-6)
        scaled_tensor = tensor / std.unsqueeze(dim).add(1e-6)
        return scaled_tensor.view(*shape)


def gradient_checkpoint(tensor: Tensor) -> GradientCheckpoint:
    """
    Gradient checkpointing to save compute for common part of graph.

    Usage:

    >>> checkpoint = gradient_checkpoint(images)
    >>> for text_loss in text_losses:
    >>>     text_loss(checkpoint.tensor()).backward()
    >>> checkpoint.continue_backward()
    """
    return GradientCheckpoint(tensor)


def test_gradient_checkpoint():
    import torch

    with torch.enable_grad():
        images = torch.zeros(1, 3, 64, 64).requires_grad_()
        checkpoint = gradient_checkpoint(images * 2)
        checkpoint.tensor().pow(2).mean().backward()
        assert checkpoint.detached.grad is not None
        checkpoint.continue_backward()
    assert images.grad is not None
