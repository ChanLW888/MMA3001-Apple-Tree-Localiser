# Overview

The branch component predicts binary trunk/scaffold-branch foreground from RGB and grayscale depth. Branch targets describe labelled tree structure rather than apple instances or the entire canopy.

| Model | Spatial modelling | Output |
|---|---|---|
| Pixel MLP (`src/train_baseline.py`) | Independent four-feature pixel classification | One raw branch logit per pixel |
| Basic U-Net (`src/train_unet.py`) | ResNet-34 encoder and skip-feature decoder | Full input-resolution branch logits |
| Extended U-Net (`src/train_improved_unet.py`) | ASPP context, attention skips and residual decoder | Full input-resolution branch logits |

Input tensors are float32 `[B, 4, H, W]` in R/G/B/depth order. Targets and logits are `[B, 1, H, W]`. Apply sigmoid and the inclusive probability cutoff for predictions; logits themselves are not binary masks. Geometry and scaling are described in [data and preprocessing](data_and_preprocessing.md).

Architecture differences suggest different representation/computation trade-offs, but no performance ranking or timing result is established by source inspection. Compare using the same data split, preprocessing and threshold. Validation is used for checkpoint selection, not independent testing.

The separate apple workflows are documented in the repository root README. They default to ground-truth branch masks, not these models' predictions. Thus branch training and apple localisation are separate workflows, not a fully connected deployment pipeline. RF-DETR comparison uses model-generated pseudo-ground truth rather than verified apple labels.
