# Source Code (`src/`)

This directory contains the scripts used to prepare the dataset, train or run apple-segmentation models, and detect/localise apples.

> Run commands from the **repository root**, not from inside `src/`, so relative input and output paths resolve consistently.

## Scripts

| Script | Purpose | Typical command |
|---|---|---|
| `dataset_organiser.py` | Organises the source dataset into the project’s expected structure. | `python src/dataset_organiser.py` |
| `apple_predictor.py` | Runs the custom RGB/depth-based apple detection/localisation workflow. The current workflow also uses branch ground-truth masks for its region-of-interest filtering. | `python src/apple_predictor.py` |
| `apple_segmenter_v1.py` | Runs the RF-DETR apple instance-segmentation baseline. | `python src/apple_segmenter_v1.py` |
| `apple_segmenter_train.py` | Trains the RF-DETR apple instance-segmentation model. | `python src/apple_segmenter_train.py` |
| `train_pixel_mlp_documented.py` | Trains the pixel-wise RGB-D MLP for binary tree-branch segmentation. | See the script’s `--help` output. |
| `resnet34_unet_train.py` | Trains the basic ResNet-34 U-Net for binary tree-branch segmentation. | See the script’s `--help` output. |
| `improved_resnet34_unet_train.py` | Trains the improved ResNet-34 U-Net with ASPP, attention-gated skip connections, and residual decoder blocks. | See the script’s `--help` output. |

The exact scripts present in a checkout may differ as the project evolves. Check the repository before running a command, and use each script’s `--help` option where supported.

## Branch-segmentation training

The three branch-segmentation models use RGB plus a grayscale depth channel as a four-channel input and predict a binary branch mask.

Example commands (run from the repository root):

```bash
python src/train_pixel_mlp_documented.py --help
python src/resnet34_unet_train.py --help
python src/improved_resnet34_unet_train.py --help
```

The training scripts require a dataset root. Supply the path using `--data-root` when launching a training run. For example:

```bash
python src/resnet34_unet_train.py --data-root /path/to/Organised
```

Replace `/path/to/Organised` with the actual path to your dataset. Do not commit private, large, or restricted dataset files to Git unless the project explicitly permits it.

### Model differences

- **Pixel-wise MLP:** predicts each pixel from its RGB-D values independently; it does not explicitly model neighbouring-pixel relationships.
- **Basic ResNet-34 U-Net:** uses a ResNet-34 encoder and U-Net-style decoder with skip connections.
- **Improved ResNet-34 U-Net:** adds multi-scale ASPP context, attention-gated skips, and residual decoder blocks.

The scripts use their own configured defaults. Refer to the code and `--help` output for the exact hyperparameters and available options.

## Outputs

Depending on the script and its arguments, generated files may include model checkpoints, CSV metrics, training history, visualisation images, and detection results. Known output locations used by the existing workflows include:

- `outputs/apple_detection/`
- `outputs/apple_rfdetr/`
- `outputs/rfdetr_apple_tree/checkpoint_best_total.pth`

Output paths can be changed by command-line arguments or script configuration. Generated outputs should generally not be committed unless they are small, intentional project artefacts.

## Dataset and evaluation notes

- The branch-segmentation scripts pair RGB, depth, and ground-truth mask files using filename identifiers.
- The training workflow uses a shuffled train/validation split; the validation set is not an independent test set.
- Winter images have been used for qualitative inference, but no winter ground-truth branch masks were available in the described workflow.
- The custom apple detector uses branch ground-truth masks for its region-of-interest filter; it should not be described as a fully predicted-branch-mask pipeline unless that part of the implementation is changed.
- The detector’s depth input is used as a validity filter in the described version; its localisation output is 2D rather than a reconstructed 3D apple position.

## Development conventions

- Keep preprocessing and checkpoint formats consistent between training and inference.
- Document any change to model inputs, mask thresholds, file naming, or output paths.
- Avoid committing datasets, checkpoints, generated plots, virtual environments, or cache files unless required.
- When changing a script, update this file and the root README if its purpose, invocation, or outputs change.
