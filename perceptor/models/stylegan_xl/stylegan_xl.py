from readline import set_completer_delims
import sys
from pathlib import Path
import dill
import numpy as np
import torch
import torch.nn as nn
from basicsr.utils.download_util import load_file_from_url

from perceptor import utils
from perceptor.models.stylegan_xl import (
    gen_utils,
    torch_utils,
    dnnlib,
    legacy,
)

checkpoints = dict(
    imagenet128="https://s3.eu-central-1.wasabisys.com/nextml-model-data/stylegan-xl/imagenet128.dill",
    ffhq256="https://s3.eu-central-1.wasabisys.com/nextml-model-data/stylegan-xl/ffhq256.dill",
    pokemon256="https://s3.eu-central-1.wasabisys.com/nextml-model-data/stylegan-xl/pokemon256.dill",
)


@utils.cache
class StyleGANXL(nn.Module):
    def __init__(self, name="imagenet128"):
        """
        Args:
            name (str, optional): Name of the model. imagenet128, ffhq256, pokemon256
        """
        super().__init__()
        self.name = name

        checkpoint_path = load_file_from_url(
            checkpoints[self.name],
            "models",
        )

        # hack for dill/pickle
        sys.modules["torch_utils"] = torch_utils
        sys.modules["dnnlib"] = dnnlib
        sys.modules["legacy"] = legacy

        self.model = dill.load(Path(checkpoint_path).open("rb"))
        self.model.eval().requires_grad_(False)

        # TODO: half precision is supported

    @property
    def device(self):
        return next(iter(self.parameters()))

    def forward(self, latents):
        return gen_utils.w_to_img(self.model, latents).add(1).div(2)

    def latents(self, size, seeds=None, class_indices=None):
        if class_indices is None:
            class_indices = [None for _ in range(size)]

        return torch.cat(
            [
                gen_utils.get_w_from_seed(
                    self.model,
                    1,
                    self.device,
                    truncation_psi=1,
                    seed=(None if seeds is None else seeds[i]),
                    class_idx=(None if class_indices is None else class_indices[i]),
                )
                for i in range(size)
            ]
        )
