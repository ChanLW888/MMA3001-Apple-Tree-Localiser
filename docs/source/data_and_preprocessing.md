# Data and preprocessing

## Dataset pairing

The branch-segmentation scripts pair RGB, depth, and ground-truth mask files
using identifiers derived from their filenames. A sample is used for training
when all three modalities are available for the same identifier.

Filename matching does not independently prove that images are spatially
aligned. Confirm that RGB, depth, and mask refer to the same scene and framing.

## Known dataset organisation

The dataset check previously reported:

| Season | RGB | Depth | Ground-truth masks | Total files |
|---|---:|---:|---:|---:|
| Summer | 562 | 562 | 562 | 1,686 |
| Winter | 562 | 562 | 0 | 1,124 |
| **Total** | **1,124** | **1,124** | **562** | **2,810** |

The branch-mask training workflow therefore uses the summer samples with
ground-truth masks. Winter images can be used for qualitative inference, but
without winter ground-truth masks they cannot provide quantitative winter
segmentation metrics.

## Image dimensions

The dataset check reported RGB and depth images at 640 × 480 pixels and
ground-truth masks at 256 × 256 pixels.

The documented preprocessing for the branch-training scripts is:

1. Centre-crop RGB and depth to a square (480 × 480 for 640 × 480 images).
2. Resize RGB to the model input size using bilinear interpolation.
3. Resize depth using nearest-neighbour interpolation.
4. Normalise RGB using ImageNet mean and standard deviation for ResNet-based
   models; scale the grayscale depth channel to [0, 1].
5. Threshold the ground-truth mask to produce a binary target.

The scripts expect the ground-truth mask to match the configured model input
size. Verify that the mask represents the same centred field of view before
training; do not silently resize or crop masks without confirming alignment.

## Depth interpretation

The depth files are stored as 8-bit image data in the checked dataset. The
physical meaning and metric calibration of these pixel values are not
established by the file format alone. They should be described as a grayscale
depth channel unless the encoding has been independently verified.
