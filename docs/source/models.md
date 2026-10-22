# Model architectures

## Pixel MLP

`src/train_baseline.py` applies dense widths 4 ->64 ->32 ->16 ->1 with ReLU between hidden layers by default. Pixels are permuted/flattened into a feature batch, classified independently, then restored to `[B, 1, H, W]`. Reshaping preserves pixel correspondence; it does not give the network neighbouring-pixel context. Hidden widths can be configured with `--hidden-dims`.

## Basic ResNet-34 U-Net

`src/train_unet.py` uses a ResNet-34 stem and four encoder stages, then bilinear resizing and concatenated skips. A padded pair of convolution/batch-normalisation/ReLU layers processes each decoder stage. Final interpolation restores the input grid, and a 1x1 convolution yields one raw branch logit per pixel.

Default ImageNet initialisation retains the RGB filters in the first convolution and uses their channel mean for depth. The implementation is intended for four-channel input; pretrained initialisation assumes RGB-D despite the constructor's configurable channel argument. `--no-pretrained` uses local random initialisation. Default pretraining may retrieve weights if not cached.

## Extended ResNet-34 U-Net

`src/train_improved_unet.py` adds:

- ASPP: a 1x1 branch plus 3x3 branches with dilation 2, 4 and 6. Concatenation/projection produces 256 context channels at the deepest encoder grid, with dropout 0.10.
- Attention gates: projected decoder context and encoder features produce a sigmoid spatial map. Multiplication preserves each skip's channel count; a gate output is a feature tensor, not a one-channel segmentation logit.
- Residual decoding: resize decoder context to the skip grid, concatenate, apply convolutions and add an identity or projected shortcut. Decoder widths are 192, 128, 96 and 64, followed by refinement and a one-channel output head.

All models output logits. Sigmoid is applied for thresholded validation and displays; BCE-with-logits consumes raw scores directly. Intermediate convolution, ASPP and decoder outputs retain their configured feature channels rather than becoming branch predictions.

The extended trainer also uses paired horizontal/vertical flips and RGB brightness/contrast perturbation after normalisation. Depth is not colour-jittered. Optimiser/loss differences are detailed in [training and evaluation](training_and_evaluation.md). These changes express design intent, not verified performance improvement.

`src/winter_inference.py` reconstructs matching model structures/state keys with torchvision weights=None and strictly loads local checkpoints. It requires compatible architectures and trusted checkpoint files; it is not a general checkpoint converter.
