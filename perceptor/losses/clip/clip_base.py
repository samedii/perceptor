import torch
from torchvision import transforms
from clip import tokenize, load


def normalize(img, input_range=None):
    if input_range is None:
        minv = img.min()
    else:
        minv = input_range[0]
    img = img - minv

    if input_range is None:
        maxv = img.max()
    else:
        maxv = input_range[1] - minv

    if maxv != 0:
        img = img / maxv

    return img


def adjust_range(img, out_range, input_range=None):
    img = normalize(img, input_range=input_range)
    img = img * (out_range[1] - out_range[0])
    img = img + out_range[0]
    return img


class CLIP_Base(torch.nn.Module):
    def __init__(self, model, preprocess, device):
        super().__init__()
        self.device = device
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.input_resolution = self.model.visual.input_resolution
        self.output_dim = self.model.visual.output_dim

        self.preprocess_transform = transforms.Compose(
            [
                transforms.Resize(self.input_resolution),
                transforms.CenterCrop(self.input_resolution),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def preprocess(self, imgs, input_range=None):
        imgs = adjust_range(imgs, [0.0, 1.0], input_range=input_range)
        return self.preprocess_transform(imgs)

    def encode_image(self, imgs, input_range=None, apply_preprocess=True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range=None)
        img_embeddings = self.model.encode_image(imgs)
        return img_embeddings / img_embeddings.norm(dim=-1, keepdim=True).float()

    def encode_text(self, text):
        text = tokenize(text).to(self.device)
        return self.model.encode_text(text).float()

    def encode_texts(self, texts):
        text_embeddings = torch.stack(
            [
                self.model.encode_text(tokenize(text).to(self.device))
                .detach()
                .clone()
                .float()
                for text in texts
            ]
        )
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


def get_clip_perceptor(clip_model_name, device):
    perceptor, preprocess = load(clip_model_name, download_root="models")
    perceptor = perceptor.requires_grad_(False).eval().to(device)

    n_params = sum(p.numel() for p in perceptor.parameters())
    in_res = perceptor.visual.input_resolution
    print(
        f"Loaded CLIP {clip_model_name}: {in_res}x{in_res} and {n_params/1000000:.2f}M params"
    )
    clip_perceptor = CLIP_Base(perceptor, preprocess, device)

    return clip_perceptor
