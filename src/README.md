# Source scripts

Run commands from the repository root. Each module uses Google-style docstrings. The [root guide](../README.md) provides setup, inputs, full invocations, limitations and testing instructions.

| Script | Responsibility | Invocation |
|---|---|---|
| `dataset_organiser.py` | Copy flat orchard PNGs into season/modality folders; basic PNG checks | `python src/dataset_organiser.py` |
| `preprocess_roboflow_dataset.py` | Merge COCO apple-* categories; export becomes valid | `python src/preprocess_roboflow_dataset.py` |
| `train_baseline.py` | Pixel-wise RGB-D MLP branch segmentation | `python src/train_baseline.py --data-root data/Organised` |
| `train_unet.py` | Basic RGB-D ResNet-34 U-Net | `python src/train_unet.py --data-root data/Organised` |
| `train_improved_unet.py` | ASPP, attention and residual U-Net; AdamW | `python src/train_improved_unet.py --data-root data/Organised` |
| `apple_predictor_v1.py` | Red-component detection with depth validity and branch GT | `python src/apple_predictor_v1.py` |
| `apple_segmenter_train.py` | RF-DETR medium apple COCO training, 50 epochs/batch 1 | `python src/apple_segmenter_train.py` |
| `apple_segmenter_v1.py` | RF-DETR apple inference with GT or externally predicted branch masks | `python src/apple_segmenter_v1.py` |
| `compare_apple_detectors.py` | Read saved predictions and measure RF-DETR-relative agreement | `python src/compare_apple_detectors.py --max-distance 30` |

The branch trainers and comparison accept `--help`. Apple inference, COCO preparation, dataset organisation and RF-DETR training use constants rather than an argparse interface. Importing the organiser copies files; importing COCO preparation/RF-DETR training checks dataset paths. Do not import these as passive utility modules.

Branch input is `[B, 4, H, W]`: normalised RGB and grayscale depth/255. Targets are `[B, 1, H, W]` binary branches, not apple masks. GT must already match the centred input geometry. Models return logits. All branch trainers select checkpoints by validation Dice; validation is not independent testing. The MLP has no spatial context and uses weighted BCE; U-Nets use plain BCE.

Apple detections are original-image 2D pixel coordinates. Traditional detection uses GT >0/full-frame resizing; RF-DETR defaults to GT >127/central-square placement. Their shared dilation size does not make their regions identical. Predicted RF-DETR branch mode only reads externally exported 0/255 PNGs; it does not run a branch model. Confidence is blank for traditional detections and a model score for RF-DETR. RF-DETR pseudo-ground truth is not human-verified evaluation annotation.

See [output formats](../outputs/README.md), [data requirements](../data/README.md) and [branch architecture guide](../docs/source/models.md). Winter inference lives in [test/winter_inference.py](../test/winter_inference.py); its output names do not directly satisfy the RF-DETR predicted-mask convention.
