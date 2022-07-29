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
  - Latent diffusion (finetuned, text2image, super resolution)
  - Guided diffusion (openai, pixelart)
  - k-diffusion (monsters, tinyhero)
- StyleGAN XL
- RuDALLE
- Deep image prior
- Super resolution
- Losses
  - Aesthetic visual assessment loss
  - BLIP
  - CLIP
  - CLOOB
  - LPIPS loss
  - Glide CLIP
  - Memorability loss
  - Midas depth
  - OpenCLIP
  - RuCLIP
  - Simulacra aesthetic loss
  - SLIP
  - Smoothness loss
  - Style transfer loss
