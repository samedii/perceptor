import torch
import lantern


class Conditioning(torch.nn.Module):
    def __init__(self, encodings):
        super().__init__()
        self.encodings = torch.nn.Parameter(encodings, requires_grad=False)

    @property
    def device(self):
        return self.encodings.device

    def __neg__(self):
        return Conditioning(-self.encodings)
