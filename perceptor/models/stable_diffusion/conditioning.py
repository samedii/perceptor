from typing import Optional
import torch
import lantern


class Conditioning(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        encodings: lantern.Tensor,
        inpainting_latent_masks: Optional[lantern.Tensor.dims("NCHW")] = None,
        inpainting_latents: Optional[lantern.Tensor.dims("NCHW")] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.encodings = torch.nn.Parameter(encodings, requires_grad=False)
        self.inpainting_latent_masks = inpainting_latent_masks
        self.inpainting_latents = inpainting_latents

    @property
    def device(self):
        return self.encodings.device

    def __neg__(self):
        return Conditioning(
            -self.encodings,
            inpainting_latent_masks=self.inpainting_latent_masks,
            inpainting_latents=self.inpainting_latents,
        )

    def input(self, diffused_latents):
        if self.model_name == "runwayml/stable-diffusion-inpainting":
            return torch.cat(
                [
                    diffused_latents,
                    self.inpainting_latent_masks.ge(0.5).float(),
                    self.inpainting_latents,
                ],
                dim=1,
            )
        else:
            return diffused_latents
