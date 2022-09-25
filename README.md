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

## API

Shortlist of available features.

```python
perceptor
  .drawers
    .BruteDiffusion
    .DeepImagePrior
    .JPEG
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
