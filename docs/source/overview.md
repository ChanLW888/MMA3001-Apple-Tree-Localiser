# Overview

## Purpose

The branch-segmentation component predicts a binary mask of the Y-shaped trunk
and primary scaffold branches from RGB and depth imagery. The target mask
represents branch structure rather than the full tree canopy.

The segmentation experiments compare three architectures with different levels
of spatial modelling and computational complexity.

## Input and output

- **Input:** RGB image plus one grayscale depth channel (four channels total).
- **Output:** one-channel binary branch-mask prediction.
- **Target:** ground-truth mask thresholded to foreground/background.
- **Reported metrics:** Dice, intersection over union (IoU), precision, and recall.

## Model comparison

| Model | Main idea | Main trade-off |
|---|---|---|
| Pixel-wise MLP | Classifies each pixel independently from its four input values. | Fast inference, but no explicit spatial context. |
| Basic ResNet-34 U-Net | Pretrained ResNet-34 encoder with a U-Net decoder and skip connections. | More spatial representation capacity and moderate computational cost. |
| Improved ResNet-34 U-Net | Adds ASPP context, attention-gated skip connections, and residual decoder blocks. | Stronger feature modelling, with additional computation. |

Performance should be compared using the same validation split, input
preprocessing, and probability threshold.
