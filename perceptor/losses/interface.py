from torch import nn


class LossInterface(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
