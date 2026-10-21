# Generated output formats

Run scripts from the repository root; see [workflow commands](../README.md). The directories below describe current source defaults, not evidence that an experiment ran successfully. Existing historical directories may have different names. Generated data, checkpoints and figures are excluded from documentation edits and Git; this README is maintained text.

## Branch training

| Trainer | Default folder | Best checkpoint |
|---|---|---|
| `src/train_baseline.py` | `outputs/pixel_mlp_crop_trainval` | `pixel_mlp_best.pth` |
| `src/train_unet.py` | `outputs/resnet34_unet_crop_trainval_output` | `resnet34_unet_best.pth` |
| `src/train_improved_unet.py` | `outputs/improved_resnet34_unet_output` | `improved_resnet34_unet_best.pth` |

Each also writes `history.csv` (epoch, training loss and validation metrics), `val_metrics.csv` (final best-state validation summary), and `visualisations/<key>_comparison.png`. Metrics include loss, Dice, IoU, precision and recall, aggregated across validation pixels with epsilon-stabilised denominators. Loss is sample-weighted. Checkpoints contain model state, input size, input channel order, mask cutoff and best validation metrics; MLP also stores hidden widths. Validation is used for model selection and is not independent testing.

Panels show RGB, depth, GT, prediction and overlay: yellow is branch overlap, green missed GT, red false predicted branch. Default probability cutoff is >=0.5. Outputs are square branch-grid coordinates, not full original-frame apple coordinates.

## Winter inference

`test/winter_inference.py` defaults to `outputs/winter_inference_benchmark`. Subfolders `pixel_mlp`, `basic_unet` and `improved_unet` contain `<key>_pred_mask.png` (0/255), orange overlays and comparison panels. Root files are:

- `per_image_timings.csv`: model, frame key, input paths, forward time in ms, input size, probability cutoff and device.
- `summary.csv`: model-only latency statistics and images/second; sample standard deviation uses ddof=1 when possible.
- `benchmark_metadata.json`: supplied checkpoint paths, size, threshold, warmup, device and timing interpretation.

Warmup is untimed. CUDA synchronisation brackets the forward pass; preprocessing, transfer, sigmoid, thresholding and disk saving are excluded. No GT-dependent winter accuracy is computed. These filenames are not directly compatible with RF-DETR's optional predicted-branch input names.

## Traditional detection

`src/apple_predictor_v1.py` writes to `outputs/apple_detection`:

- `<original-RGB-basename>.png` annotations directly in the folder, with orange boxes/centres/indices and apple count.
- `apple_counts.csv`: `Tree,Season,Time,Apple_Count,Annotated_Image`.
- `apple_instances.csv`: `image,apple_id,confidence,area_px,branch_overlap,cx_px,cy_px,x1,y1,x2,y2,red_score,image_width,image_height`.
- `image_timings.csv`: `image,apple_count,read_ms,detection_ms,save_ms,total_ms`.
- `run_metadata.json`: algorithm label, GT branch source, full-resize alignment, coordinate conventions and detector parameters.

Traditional confidence is blank; red_score is mean component R/(G+B+1), not confidence. area_px is cleaned-component pixel area. branch_overlap is the fraction of that component within the dilated GT branch zone. Centroids are component mean pixel coordinates. Read time includes loading/GT alignment; detection includes masking/filtering; save includes annotation rendering/writing. Total includes per-frame wrapper overhead and can exceed their sum. CSV/metadata export after frame processing is outside those frame totals.

## RF-DETR

Training (`src/apple_segmenter_train.py`) writes under `outputs/rfdetr_apple_tree`; inference expects `checkpoint_best_total.pth`. Other training artefacts are delegated to the installed RF-DETR library; they are not enumerated as guaranteed here.

`src/apple_segmenter_v1.py` defaults to `outputs/apple_rfdetr_instances`:

- `annotated/<RGB-stem>_annotated.png`: cyan boxes, centres, indices and count; no coloured instance overlay.
- `masks/<RGB-stem>_instances.png`: uint16 IDs, background 0, kept apples 1 onwards. Later overlapping masks overwrite earlier IDs; CSV areas still refer to whole instances.
- `branch_zones/<RGB-stem>_zone.png`: dilated binary zone encoded 0/255.
- `apple_counts.csv`: `image,tree,season,time,branch_source,rf_detr_detections,kept_apple_count`.
- `apple_instances.csv`: `image,apple_id,confidence,area_px,branch_overlap,cx_px,cy_px,x1,y1,x2,y2`.
- `image_timings.csv`: `image,rf_detr_detections,kept_apple_count,read_ms,inference_ms,filter_ms,save_ms,total_ms`.

Raw detection count precedes branch/area filtering; kept count follows it. Confidence is model output, not verified probability calibration. Centroids and boxes derive from whole binary masks. Inference time synchronises CUDA and surrounds model.predict; model construction is excluded. Read includes mask alignment/zone creation; saving includes image/row export. These timing boundaries differ from winter model-only forward timing. Comparison does not analyse runtime.

Both apple instance CSVs use original RGB pixels: top-left origin, x right, y down; boxes are inclusive xyxy, centres may be fractional. Apple IDs start at 1 per image. Area is in pixels, branch_overlap in [0, 1]. Counts CSVs retain processed images with no detections; an instance CSV alone cannot distinguish zero from unprocessed.

Named CSVs are overwritten each run. `START_FRAME >1` recreates RF-DETR CSVs for selected frames; it does not append or remove stale PNGs. GT masks and externally predicted masks have different provenance; defaults use GT. Current source settings do not establish settings used for historical CSVs.

## Detector comparison

`src/compare_apple_detectors.py` writes `outputs/apple_detector_comparison` by default:

| File | Meaning |
|---|---|
| `image_coverage.csv` | Exact basename union, marked matched/reference_only/traditional_only |
| `duplicate_detections.csv` | Repeated geometry retained in inputs; duplicate IDs instead raise errors |
| `per_image_results.csv` | Counts, signed/absolute error and reference-relative agreement; unmatched images have undefined cross-method fields |
| `summary_metrics.csv` | Count MAE/RMSE, exact/within-one rates, global agreement, conditional distance/IoU and macro image F1 |
| `matched_instances.csv` | Matched/unmatched IDs, accepted centroid distance in px, inclusive-box IoU and status |
| `threshold_sensitivity.csv` | Repeats matching at supplied tolerances with fixed detector outputs |
| `input_manifest.json` | SHA-256 of available CSV/source inputs, arguments and source-image dimensions |
| `annotation_inventory.json` | Candidate local COCO JSON counts/categories; does not verify annotation authorship |
| `evaluation_report.md` | Generated interpretation and fixed source-description prose |
| `count_comparison.png`, `count_error_distribution.png`, `centroid_error_distribution.png`, `threshold_sensitivity.png` | Agreement/sensitivity plots |
| `comparison_examples/` | Selected agreement, overcount, undercount and localisation cases; cyan RF-DETR, orange traditional, white match links |

Positive signed count error means traditional exceeds RF-DETR. Undefined precision/recall/F1 ratios become zero; no matched distances/IoUs yield blank CSV values. Statistics use common processed frames including zero-count images; unmatched images are listed separately. RF-DETR is pseudo-ground truth. Different GT mask alignment/cutoffs, depth filtering and possible merged/split instances confound detector-only attribution. Conditional matching distances do not establish true localisation error.

The generated report contains fixed historical statements (including an obsolete `MIN_RED_AREA` remark and claims about tuning/reruns) that are not proven by its inputs. They remain executable string data and were not edited by this documentation task. Interpret source defaults and manifests as available evidence, not certification of historical experimental provenance.
