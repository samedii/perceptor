from lantern import FunctionalBase, Tensor


class GradientCheckpoint(FunctionalBase):
    original: Tensor
    detached: Tensor

    def __init__(self, tensor):
        super().__init__(original=tensor, detached=tensor.detach().requires_grad_())

    def continue_backward(self, retain_graph=False):
        if self.grad is None:
            raise ValueError("Gradient is not defined")
        return self.original.backward(self.detached.grad, retain_graph=retain_graph)

    @property
    def grad(self):
        return self.detached.grad

    def tensor(self):
        return self.detached


def gradient_checkpoint(tensor: Tensor) -> GradientCheckpoint:
    """
    Gradient checkpointing to save compute for common part of graph.

    Usage:

        checkpoint = gradient_checkpoint(images)
        for text_loss in text_losses:
            text_loss(checkpoint.tensor()).backward()
        checkpoint.continue_backward()
    """
    return GradientCheckpoint(tensor)


def test_gradient_checkpoint():
    import torch

    with torch.enable_grad():
        images = torch.zeros(1, 3, 64, 64).requires_grad_()
        checkpoint = gradient_checkpoint(images * 2)
        checkpoint.tensor().pow(2).mean().backward()
        assert checkpoint.grad is not None
        checkpoint.continue_backward()
    assert images.grad is not None
