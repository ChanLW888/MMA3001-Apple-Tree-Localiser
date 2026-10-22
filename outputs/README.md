# Generated Output Formats

This directory contains generated model checkpoints, predictions, annotations, evaluation results and optimisation outputs. Run all scripts from the repository root. Generated files are excluded from Git unless explicitly included.

## 1. Branch Segmentation Training

| Script | Output directory | Best checkpoint |
|---|---|---|
| `src/train_baseline.py` | `outputs/pixel_mlp_crop_trainval/` | `pixel_mlp_best.pth` |
| `src/train_unet.py` | `outputs/resnet34_unet_crop_trainval/` | `resnet34_unet_best.pth` |
| `src/train_improved_unet.py` | `outputs/improved_resnet34_unet/` | `improved_resnet34_unet_best.pth` |

Each training directory contains:

- `history.csv` — Training loss and validation metrics across epochs.
- `val_metrics.csv` — Best-checkpoint validation results.
- `visualisations/` — Predicted masks, ground-truth comparisons and overlays.
- `*.pth` — Saved model checkpoint.

## 2. Winter Branch Inference

**Script:** `test/winter_inference.py`

**Output:** `outputs/winter_inference_benchmark/`

```text
winter_inference_benchmark/
├── pixel_mlp/
├── basic_unet/
├── improved_unet/
├── per_image_timings.csv
├── summary.csv
└── benchmark_metadata.json
```

Each model folder contains predicted branch masks, annotated overlays and comparison images. The CSV files record inference timing and summary statistics.

## 3. Traditional Apple Detection

**Script:** `src/apple_predictor_v1.py`

**Output:** `outputs/apple_detection/`

```text
apple_detection/
├── <original-RGB-basename>.png
├── apple_counts.csv
├── apple_instances.csv
├── image_timings.csv
└── run_metadata.json
```

| File | Contents |
|---|---|
| Annotated PNGs | Original RGB images with orange bounding boxes, centroids, detection IDs and apple counts |
| `apple_counts.csv` | Tree, season, time, apple count and annotated image path |
| `apple_instances.csv` | Individual apple IDs, centroids, bounding boxes, area, branch overlap and red-dominance score |
| `image_timings.csv` | Per-image reading, detection, saving and total processing times |
| `run_metadata.json` | Detector parameters, branch source and alignment settings |

## 4. RF-DETR Apple Detection

**Training script:** `src/apple_segmenter_train.py`

**Inference script:** `src/apple_segmenter_v1.py`

**Training output:** `outputs/rfdetr_apple_tree/`

**Inference output:** `outputs/apple_rfdetr_instances/`

```text
apple_rfdetr_instances/
├── annotated/
├── masks/
├── branch_zones/
├── apple_counts.csv
├── apple_instances.csv
└── image_timings.csv
```

| File / Folder | Contents |
|---|---|
| `annotated/` | RGB images annotated with cyan bounding boxes, centroids, IDs and apple counts |
| `masks/` | Instance-segmentation masks with individual apple IDs |
| `branch_zones/` | Dilated branch-region masks |
| `apple_counts.csv` | Per-image raw and branch-filtered apple counts |
| `apple_instances.csv` | Individual detections, confidence scores, areas, centroids and bounding boxes |
| `image_timings.csv` | Per-image inference, filtering and processing times |

RF-DETR inference uses the trained checkpoint `outputs/rfdetr_apple_tree/checkpoint_best_total.pth`.

## 5. Apple Detector Comparison

**Script:** `src/compare_apple_detectors.py`

**Output:** `outputs/apple_detector_comparison/`

| File / Folder | Contents |
|---|---|
| `summary_metrics.csv` | Overall detection, counting and localisation metrics |
| `per_image_results.csv` | Per-image comparison results |
| `matched_instances.csv` | Matched and unmatched apple detections |
| `image_coverage.csv` | Images included or excluded from comparison |
| `duplicate_detections.csv` | Duplicate detection information |
| `threshold_sensitivity.csv` | Results across different centroid-matching distances |
| `input_manifest.json` | Input files, parameters and checksums |
| `annotation_inventory.json` | Available annotation dataset information |
| `evaluation_report.md` | Generated comparison summary |
| `count_comparison.png` | Traditional versus RF-DETR apple counts |
| `count_error_distribution.png` | Distribution of counting differences |
| `centroid_error_distribution.png` | Matched-centroid distance distribution |
| `threshold_sensitivity.png` | Evaluation matching-threshold sensitivity |
| `comparison_examples/` | Visual comparisons between both detectors |

The comparison uses saved detector outputs, with RF-DETR predictions serving as pseudo-ground-truth references rather than independently verified annotations.

## 6. Apple Detector Grid-Search Optimisation

**Script:** `src/full_grid_search_apple.py`

**Output:** `outputs/apple_grid_search/`

The grid search evaluates combinations of traditional apple-detector parameters while keeping branch-region filtering and RF-DETR reference predictions fixed.

```text
apple_grid_search/
├── all_experiments.csv
├── best_by_metric.csv
├── metric_rankings.csv
├── heldout_comparison.csv
└── grid_metadata.json
```

| File | Contents |
|---|---|
| `all_experiments.csv` | Parameter combinations and their precision, recall, F1-score, counting errors, localisation metrics and runtime |
| `best_by_metric.csv` | Best-performing parameter configurations for individual evaluation metrics |
| `metric_rankings.csv` | Ranked configurations for each optimisation objective |
| `heldout_comparison.csv` | Baseline versus selected configuration on held-out trees |
| `grid_metadata.json` | Grid-search settings, data split and reproducibility information |

Run the full grid search using:

```powershell
conda activate mma3001
python src/full_grid_search_apple.py
```

The optimisation varies colour thresholds, component-area limits, circularity, aspect ratio and morphology size. Branch dilation and branch-overlap settings remain unchanged.

## Notes

- Detector outputs are stored in original RGB-image pixel coordinates.
- Generated CSV files may be overwritten when scripts are rerun.
- Model checkpoints, datasets and generated results are maintained locally unless explicitly committed.
- RF-DETR comparison metrics measure agreement with pseudo-ground truth, not absolute apple-detection accuracy.
