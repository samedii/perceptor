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
    .JPEG
    .Raw
    .RuDALLE
    .DeepImagePrior
    .StyleGANXL
  .models
    .VelocityDiffusion (yfcc_2, yfcc_1, cc12m_1_cfg, wikiart)
    .ldm (latent diffusion)
      .FinetunedText2Image
      .Text2Image
      .Face
      .SuperResolution
    .GuidedDiffusion (openai, pixelart)
    .MonsterDiffusion (all, tinyhero)
    .StyleGANXL
    .RuDALLE
    .DeepImagePrior
    .SuperResolution
  .losses
    # Text-image similarity
    .BLIP
    .CLIP
    .CLOOB
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
