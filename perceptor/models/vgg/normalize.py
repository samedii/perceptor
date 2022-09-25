import torch


def normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(image.device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image
