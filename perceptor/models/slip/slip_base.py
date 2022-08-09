from collections import OrderedDict
import torch
from basicsr.utils.download_util import load_file_from_url
from torchvision import transforms

from perceptor.transforms import resize
from . import models
from .tokenizer import SimpleTokenizer

slip_models_ckpt = {
    "SLIP_VITS16": "slip_small_100ep.pt",
    "SLIP_VITB16": "slip_base_100ep.pt",
    "SLIP_VITL16": "slip_large_100ep.pt",
    "CLIP_VITS16": "clip_small_25ep.pt",
    "CLIP_VITB16": "clip_base_25ep.pt",
    "CLIP_VITL16": "clip_large_25ep.pt",
    "SLIP_CC3M": "slip_base_cc3m_40ep.pt",
    "SLIP_CC12M": "slip_base_cc12m_35ep.pt",
}


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


class SLIP_Base(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.input_resolution = 224
        ckpt_file = slip_models_ckpt[model_name]
        ckpt_path = f"models/{ckpt_file}"
        load_file_from_url(
            f"https://dl.fbaipublicfiles.com/slip/{ckpt_file}",
            "models",
        )

        self.preprocess_transform = transforms.Compose(
            [
                # transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.tokenizer = SimpleTokenizer()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v

        # create model
        old_args = ckpt["args"]
        old_args.model = model_name
        # these two are the same model on different training data...
        if old_args.model == "SLIP_CC3M" or old_args.model == "SLIP_CC12M":
            old_args.model = "SLIP_VITB16"

        self.model = getattr(models, old_args.model)(
            rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim,
            ssl_emb_dim=old_args.ssl_emb_dim,
        )
        self.model.requires_grad_(False).eval()
        self.model.load_state_dict(state_dict, strict=True)
        n_params = sum(p.numel() for p in self.model.parameters())
        print("Loaded perceptor %s: %.2fM params" % (model_name, (n_params / 1000000)))

    def preprocess(self, imgs, input_range=None):
        imgs = adjust_range(imgs, [0.0, 1.0], input_range=input_range)
        return self.preprocess_transform(imgs)

    def encode_image(self, imgs, input_range=None, apply_preprocess=True):
        if apply_preprocess:
            imgs = resize(
                imgs,
                out_shape=(self.input_resolution, self.input_resolution),
            )
            imgs = self.preprocess(imgs, input_range=input_range)
        image_features = self.model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        texts = self.tokenizer(texts)
        texts = texts.view(-1, 77).contiguous()
        text_embeddings = self.model.encode_text(texts)
        return text_embeddings

    def encode_texts(self, texts):
        texts = self.tokenizer(texts)
        texts = texts.view(-1, 77).contiguous()
        text_embeddings = self.model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.unsqueeze(1)


def get_slip_perceptor(model_name):
    return SLIP_Base(model_name)
