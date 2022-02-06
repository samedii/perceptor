from torch import nn


class TransformInterface(nn.Module):
    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError
