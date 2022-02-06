import numpy as np
import torch
from perlin_numpy import generate_fractal_noise_2d


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# https://stats.stackexchange.com/a/289477
def contrast_noise(n):
    n = 0.9998 * n + 0.0001
    n1 = n / (1 - n)
    n2 = np.power(n1, -2)
    n3 = 1 / (1 + n2)
    return n3


def fractal(shape):
    n, c, h, w = shape
    # scale up roughly as power of 2
    if w > 1024 or h > 1024:
        side, octp = 2048, 6
    elif w > 512 or h > 512:
        side, octp = 1024, 5
    elif w > 256 or h > 256:
        side, octp = 512, 4
    else:
        side, octp = 256, 3

    return torch.from_numpy(
        np.stack(
            [
                np.stack(
                    [
                        contrast_noise(
                            normalize(
                                generate_fractal_noise_2d((side, side), (32, 32), octp)
                            )
                        )[:h, :w]
                        for _ in range(c)
                    ]
                )
                for _ in range(n)
            ]
        )
    ).float()
