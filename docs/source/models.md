# Model architectures

## Pixel-wise MLP

The MLP applies a stack of fully connected layers to each pixel's four RGB-D
features independently. Its output for a pixel does not directly depend on the
neighbouring pixels' features. This makes the model simple and efficient, but
limits its ability to represent branch continuity and shape.

## Basic ResNet-34 U-Net

The basic model uses a ResNet-34 encoder and a U-Net-style decoder. Encoder
features are fused into the decoder through skip connections to retain
spatial information while progressively recovering resolution.

The first ResNet convolution is adapted from three input channels to four.
When ImageNet-pretrained weights are used, the RGB filters are retained and
the added depth-channel filters are initialised from the mean of the original
RGB filters.

## Improved ResNet-34 U-Net

The improved model adds three modules:

### Atrous Spatial Pyramid Pooling (ASPP)

Parallel convolution branches use different dilation rates to gather context
at multiple spatial scales.

### Attention-gated skip connections

Attention gates weight encoder features before they are concatenated with
decoder features. The intention is to emphasise features relevant to the
decoder's current representation.

### Residual decoder blocks

Residual shortcuts provide a direct path through decoder blocks and combine
the block input with the convolutional transformation.

The improved training workflow also uses paired horizontal and vertical flips
on training images and masks, plus mild brightness/contrast jitter on RGB
channels only. The depth channel is left unchanged by the colour augmentation.
