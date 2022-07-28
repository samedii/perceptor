# perceptor

Modular image generation library.

- Diffusion models
  - Velocity diffusion (yfcc_2, yfcc_1, cc12m_1_cfg, wikiart)
  - Latent diffusion (finetuned, text2image, super resolution)
  - Guided diffusion (openai, pixelart)
  - k-diffusion (monsters, tinyhero)
- StyleGAN XL
- RuDALLE
- Deep image prior
- Super resolution
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

## Install

```
poetry add git+https://github.com/samedii/perceptor.git
```

Or, for the old timers:

```
pip install git+https://github.com/samedii/perceptor.git
```

### CUDA 11.3 (optional)

The default version of pytorch only supports CUDA 10 so if you for example
have an RTX card then you need to switch to CUDA 11 manually.

```
poetry run python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
