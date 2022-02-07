from torch import nn


class DrawingInterface(nn.Module):
    def forward(self, *args, **kwargs):
        return self.synthesize(*args, **kwargs)

    def synthesize(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def replace_(self, *args, **kwargs):
        raise NotImplementedError
