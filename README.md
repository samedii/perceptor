# perceptor

Modular image generation library.

## Install

```bash
poetry add perceptor
```

Or, for the old timers:

```bash
pip install perceptor
```

### CUDA 11.3 (Support RTX cards)

```bash
poetry run pip uninstall torch torchvision -y && poetry run pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Interface

Shortlist of available features. See the [API reference](https://perceptor.readthedocs.io/en/latest/) for more information.

```python
perceptor
  .drawers
    .BruteDiffusion
    .DeepImagePrior
    .Raw
    .RuDALLE
    .StyleGANXL
  .models
    .AdaBinsDepth
    .DeepImagePrior
    .VelocityDiffusion (yfcc_2, yfcc_1, cc12m_1_cfg, wikiart)
    .latent_diffusion
      .Text2Image
      .Face
      .SuperResolution
    .GuidedDiffusion (openai, pixelart)
    .MidasDepth
    .MonsterDiffusion (all, tinyhero)
    .StyleGANXL
    .RuDALLE
    .StableDiffusion
    .SuperResolution
  .losses
    # Text-image similarity
    .BLIP
    .CLIP
    .CLOOB
    .LiT
    .GlideCLIP
    .OpenCLIP
    .OWLViT
    .RuCLIP
    .SLIP
    # Other
    .AestheticVisualAssessment
    .LPIPS
    .Memorability
    .MidasDepth
    .SimulacraAesthetic
    .Smoothness
    .StyleTransfer
  .transforms
    .clamp_with_grad
    .resize
    .SuperResolution
  .utils
    .gradient_checkpoint
```
