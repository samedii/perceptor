"""
Highly modified version of https://github.com/shariqfarooq123/AdaBins/blob/main/infer.py
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from . import model_io
from .models import UnetAdaptiveBins


class InferenceHelper(nn.Module):
    def __init__(self, dataset, pretrained_path):
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if dataset == "nyu":
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(
                n_bins=256, min_val=self.min_depth, max_val=self.max_depth
            )
        elif dataset == "kitti":
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(
                n_bins=256, min_val=self.min_depth, max_val=self.max_depth
            )
        else:
            raise ValueError(
                f"dataset can be either 'nyu' or 'kitti' but got {dataset}"
            )

        model, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        model.requires_grad_(False)
        self.model = model

    def predict(self, image):
        """
        Args:
            image: torch.Tensor of shape (1, 3, H, W) between 0 and 1
        """
        bins, pred = self.model(self.normalize(image))
        pred = pred.clamp(self.min_depth, self.max_depth)

        # Flip
        pred_lr = TF.hflip(
            self.model(TF.hflip(image))[-1].clamp(self.min_depth, self.max_depth)
        )

        # Take average of original and mirror
        final = (pred + pred_lr) / 2
        final = F.interpolate(
            final,
            image.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).clamp(self.min_depth, self.max_depth)

        # final[np.isinf(final)] = self.max_depth
        # final[np.isnan(final)] = self.min_depth

        # centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        # centers = centers.cpu().squeeze().numpy()
        # centers = centers[centers > self.min_depth]
        # centers = centers[centers < self.max_depth]

        # return centers, final
        return final
