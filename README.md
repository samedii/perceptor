# perceptor

Modular image generation library.

## Install

```
poetry add perceptor
```

Or, for the old timers:

```
pip install perceptor
```

## Features

- Diffusion models
  - Velocity diffusion (yfcc_2, yfcc_1, cc12m_1_cfg, wikiart)
  - Latent diffusion (finetuned, text2image, super resolution, face)
  - Guided diffusion (openai, pixelart)
  - Monster diffusion (all, tinyhero)
- StyleGAN XL
- RuDALLE
- Deep image prior
- Super resolution
- Losses
  - Text-image similarity
    - BLIP
    - CLIP
    - CLOOB
    - Glide CLIP
    - OpenCLIP
    - OWL-ViT
    - RuCLIP
    - SLIP
  - Aesthetic visual assessment loss
  - LPIPS loss
  - Memorability loss
  - Midas depth
  - Simulacra aesthetic loss
  - Smoothness loss
  - Style transfer loss
