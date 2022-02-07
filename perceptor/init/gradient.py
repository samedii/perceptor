import numpy as np
import torch


def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(
        zip(start_list, stop_list, is_horizontal_list)
    ):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result


def gradient(shape):
    n, c, h, w = shape
    if c != 3:
        raise ValueError("Only 3 channel images are supported.")

    return (
        torch.from_numpy(
            np.stack(
                [
                    gradient_3d(
                        w,
                        h,
                        (0, 0, np.random.randint(0, 255)),
                        (
                            np.random.randint(1, 255),
                            np.random.randint(2, 255),
                            np.random.randint(3, 128),
                        ),
                        (True, False, False),
                    )
                    / 255
                    for _ in range(n)
                ]
            )
        )
        .float()
        .permute(0, 3, 1, 2)
    )
