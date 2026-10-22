# Source Scripts

This directory contains the Python implementations for dataset preparation, branch segmentation, apple localisation, model comparison and parameter optimisation.

Run all commands from the repository root. Source modules use Google-style docstrings to document their functions, parameters and outputs. See the [main README](../README.md) for environment setup, dependencies and reproduction instructions.

## Script Overview

| Script | Responsibility | Invocation |
|---|---|---|
| `dataset_organiser.py` | Organise orchard PNGs into season and modality folders | `python src/dataset_organiser.py` |
| `preprocess_roboflow_dataset.py` | Prepare COCO apple annotations for RF-DETR training | `python src/preprocess_roboflow_dataset.py` |
| `train_baseline.py` | Train the pixel-wise RGB-D MLP branch-segmentation baseline | `python src/train_baseline.py --data-root data/Organised` |
| `train_unet.py` | Train the baseline ResNet-34 U-Net branch-segmentation model | `python src/train_unet.py --data-root data/Organised` |
| `train_improved_unet.py` | Train the improved U-Net with ASPP, attention gates and residual decoder blocks | `python src/train_improved_unet.py --data-root data/Organised` |
| `winter_inference.py` | Qualitative winter predictions and model-only timing for three checkpoints | Use the complete command below; dataset and three checkpoint flags are required |
| `apple_predictor_v1.py` | Traditional apple detection using colour, depth validity, geometry and ground-truth branches | `python src/apple_predictor_v1.py` |
| `apple_predictor_predicted_compare.py` | Traditional detection with improved-U-Net branches and optional aligned-GT control | `python src/apple_predictor_predicted_compare.py --help` |
| `apple_segmenter_train.py` | Train the RF-DETR apple instance-segmentation model | `python src/apple_segmenter_train.py` |
| `apple_segmenter_v1.py` | Run RF-DETR apple detection with branch-region filtering | `python src/apple_segmenter_v1.py` |
| `compare_apple_detectors.py` | Compare traditional and RF-DETR apple detections using counting, matching and localisation metrics | `python src/compare_apple_detectors.py --max-distance 30` |
| `full_grid_search_apple.py` | Perform multi-parameter grid-search optimisation of the traditional apple detector | `python src/full_grid_search_apple.py` |

## Branch Segmentation

The branch models use RGB-D inputs `[B, 4, H, W]` and produce raw logits
`[B, 1, H, W]`. Sigmoid and thresholding produce binary masks of the labelled
trunk/scaffold branches; the network output itself is not a binary mask.

Three architectures are implemented:

- **Pixel-wise MLP:** A baseline using individual pixel features without spatial context.
- **Baseline U-Net:** A ResNet-34 encoder–decoder with skip connections for spatial feature extraction.
- **Improved U-Net:** Extends the baseline with ASPP, attention-gated skip connections and residual decoder blocks.

All models are trained using summer RGB-D images and ground-truth branch masks. Training checkpoints are selected using validation Dice.

### Winter Inference

`winter_inference.py` loads the three trained models and generates branch predictions on winter RGB-D images.

It produces predicted segmentation masks, visual comparisons and inference-time measurements. Since winter ground-truth branch masks are unavailable, the results provide qualitative comparisons and runtime benchmarks rather than quantitative segmentation accuracy.

With checkpoints in the current default training locations:

```powershell
python src/winter_inference.py --data-root data/Organised --mlp-checkpoint outputs/pixel_mlp_crop_trainval/pixel_mlp_best.pth --unet-checkpoint outputs/resnet34_unet_crop_trainval_output/resnet34_unet_best.pth --improved-checkpoint outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth
```

Supply actual checkpoint paths for historical/custom runs. These files are not in Git.

## Apple Localisation

Two apple-detection approaches are implemented:

- **Traditional detector:** Identifies red-dominant connected components using depth validity, morphological processing, geometric filtering and proximity to ground-truth Y-shaped branches.
- **RF-DETR:** Performs instance segmentation and applies branch-region filtering to retain relevant apple detections.

Both approaches output apple counts, bounding boxes and centroids in original-image pixel coordinates.

The standalone implementations use GT branches by default; the predicted driver connects improved-U-Net masks to traditional detection. However, their mask-alignment methods differ: the traditional detector uses full-frame resizing, while RF-DETR uses centre-crop alignment. This difference must be considered when interpreting comparisons.

RF-DETR predictions are used as **pseudo-ground-truth references**, not independently verified apple annotations.

## Detector Comparison and Optimisation

`compare_apple_detectors.py` evaluates saved traditional and RF-DETR predictions using precision, recall, F1-score, counting MAE, RMSE, centroid distance and bounding-box IoU.

`full_grid_search_apple.py` evaluates combinations of traditional detector parameters, including colour thresholds, component-area limits, circularity, aspect ratio and morphological kernel size.

The grid search records multiple performance metrics for each configuration and supports selecting parameters using development images and evaluating the selected configuration on held-out trees.

Branch dilation, branch-overlap settings and RF-DETR reference predictions remain fixed during optimisation.

## Related Documentation

- [Main README](../README.md) — Project overview, Conda environments and reproduction instructions.
- [Data Requirements](../data/README.md) — Orchard dataset structure and RF-DETR training-data preparation.
- [Output Formats](../outputs/README.md) — Model checkpoints, detection outputs, evaluation CSVs and grid-search results.
- [Branch Architecture Guide](../docs/source/models.md) — Details of the branch-segmentation architectures.
- [HTML Documentation](../docs/html/index.html) — Generated source-code API documentation.

**Note:** Dataset organisation and RF-DETR training/preparation scripts may perform file operations or configuration checks when imported. Run them as standalone scripts rather than importing them as passive utilities.
## Full grid-search command-line options

Run `conda run -n mma3001 python src/full_grid_search_apple.py --help`.
The maintained optimiser is `full_grid_search_apple.py`; the former one-factor
optimiser is absent from this checkout.

| Option | Default / accepted values |
|---|---|
| `--predictor` | `src/apple_predictor_v1.py` |
| `--comparison` | `src/compare_apple_detectors.py` |
| `--reference-dir` | `outputs/apple_rfdetr_instances` (counts and instances CSVs) |
| `--output-dir` | `outputs/apple_grid_search` |
| `--max-distance` | 30.0 original-image pixels, inclusive |
| `--select-by` | `f1` (default), `precision`, `recall`, `mae`, `rmse`, `centroid`, `bbox_iou`, `count_bias` |
| `--split` | `by-tree` (default), `all` |
| `--max-runs` | No limit; stop after this many newly completed combinations |
| `--no-cache` | Reload the complete split into a new list each configuration; does not stream frames |

The grid has 5,184 combinations. Resume validates metadata and skips recorded run
IDs; partial rankings are not exhaustive results. Source edits change hashes.
The selection objective is not recorded in resume metadata: an existing held-out
CSV is reused even after `--select-by` changes. Use separate output directories.
Neither `--max-runs` nor `--no-cache` removes the real-data requirement.

Despite the CLI help's stated RAM-reduction intent, `--no-cache` still loads the
whole split. The previous iteration's list can remain referenced while its
replacement is built, allowing two split-sized lists at peak. No reduced-memory
claim is justified without a separate implementation change and measurement.

The exact current search space is:

| Constant | Values |
|---|---|
| `RED_THRESHOLD` | 30, 40, 50, 60 |
| `RED_DOMINANCE` | 1.2, 1.35, 1.5, 1.65 |
| `MIN_APPLE_AREA` | 10, 20, 30, 50 pixels |
| `MAX_APPLE_AREA` | 2000, 5000, 8000 pixels |
| `MIN_CIRCULARITY` | 0.10, 0.20, 0.30 |
| `MAX_ASPECT_RATIO` | 1.8, 2.5, 3.5 |
| `MORPH_KERNEL_SIZE` | 3, 5, 7 pixels (ellipse diameter) |

`select_best` prefers the exact baseline on tied objective scores, then the
lexicographically smallest run ID. The rankings CSV instead breaks ties only by
run ID; its first row can differ from `best_by_metric.csv` on a tie. Count bias
minimises absolute signed error, while MAE/RMSE/centroid minimise error and
F1/precision/recall/IoU maximise agreement. Conditional centroid/IoU objectives can
favour a small matched subset. Invalid scores normally rank last, but count-bias
NaN handling is inconsistent. The search script itself, image bytes, environment
and complete held-out filenames are not hashed. Choosing multiple objectives after
viewing held-out results would compromise that held-out interpretation.

## Software unit tests

Run `conda run -n mma3001 python -m unittest discover -s test -p "test_*.py" -v`
from the repository root. See [test coverage and limitations](../test/README.md).
Winter inference is a maintained source workflow, not a unit test.


## Predicted branches and aligned GT control

`apple_predictor_predicted_compare.py` is the summer branch-to-apple entry point.
It imports the traditional detector, lazily imports `winter_inference`/PyTorch for
fresh masks and invokes the comparator as a subprocess in the same interpreter.
It does not invoke RF-DETR. See the [complete guide](../docs/source/predicted_branches.md)
for commands, all flags, outputs, alignment, cache limitations and failure modes.
