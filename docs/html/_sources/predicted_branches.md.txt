# Predicted branches versus GT branches

`src/apple_predictor_predicted_compare.py` connects improved U-Net branch inference
to traditional apple detection. It compares exported detections to existing
RF-DETR CSVs without running RF-DETR. Run from the repository root in `mma3001`.

| Experiment | Branch source and geometry | Default detection directory |
|---|---|---|
| Standalone `apple_predictor_v1.py` | GT >0; nearest-neighbour full-frame stretch | `outputs/apple_detection` |
| Predicted driver default | Improved U-Net sigmoid >=0.5; central square placement | `outputs/apple_detection_predicted` |
| Driver GT control | GT >127; central square placement | `outputs/apple_detection_gt_aligned` |
| RF-DETR reference | Saved model predictions; standalone source defaults use GT >127 and central placement | `outputs/apple_rfdetr_instances` |

Saved reference settings must be checked independently. CSV geometry alone cannot
prove branch source, threshold or checkpoint. RF-DETR detections are pseudo-ground
truth, not human apple annotation. Evaluating summer frames used for branch training
or checkpoint selection does not establish independent branch generalisation.

## Reproduce fresh predictions

Prerequisites: registered summer RGB-D pairs, an architecture-compatible improved
U-Net checkpoint, and RF-DETR `apple_counts.csv`/`apple_instances.csv` with source
RGBs. GT is optional until a control is requested. The checkpoint can contain
`model_state_dict` plus `input_size`, or a bare state dictionary (size fallback 256).
The reconstructed model uses no pretrained downloads, strict loading and eval mode;
`torch.load(..., weights_only=True)` restricts deserialisation.

```powershell
conda activate mma3001
python src/apple_predictor_predicted_compare.py --help
python src/apple_predictor_predicted_compare.py --checkpoint outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth --also-gt-control
```

The checkpoint path above is the current trainer default, not a bundled file.
Historical local runs may use `outputs/improved_resnet34_unet` instead. Supply the
actual path. For a small intentional run, use `--max-images 2` and fresh output/mask
directories; this limits frames, not the interpretation requirements. For detection
without saved references, add `--skip-comparison`.

RGB/depth are centre-cropped independently to their largest squares and resized to
the checkpoint input side. RGB is bilinear, /255 and ImageNet standardised; L-mode
depth is nearest-neighbour and /255. This depth channel is intensity, not calibrated
distance. A float32 `[1,4,S,S]` batch yields logits `[1,1,S,S]`; sigmoid and inclusive
thresholding yield a square 0/255 PNG. The mask is nearest-neighbour expanded to
`min(H,W)`, placed at `((W-side)//2,(H-side)//2)` on a zero canvas, then dilated by the
unchanged traditional detector (ellipse diameter 100, overlap >=0.05).

Apples are detected on the full original RGB/depth grids. Centroids and inclusive
xyxy boxes remain in original pixels, with x right and y down. Predicted masks are
not GT and are not full-frame apple labels. Alignment assumes GT describes the
same central field of view; matching dimensions do not establish registration.
The projection helper stretches nonsquare 2D masks; the cache reader rejects
nonsquare masks. RF-DETR's same-size-mask shortcut differs from the driver's
unconditional centre projection, so equivalence requires the intended square inputs.

## CLI contract

| Option | Default / behaviour |
|---|---|
| `--season` | `Summer`; does not automatically change directory defaults |
| `--rgb-dir`, `--depth-dir`, `--gt-dir` | `data/Organised/Summer/RGB`, `D`, `GT` |
| `--checkpoint` | `outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth` |
| `--reference-dir` | `outputs/apple_rfdetr_instances` |
| `--output-dir` | `outputs/apple_detection_predicted` |
| `--comparison-dir` | `outputs/apple_detector_comparison_predicted` |
| `--gt-output-dir` | `outputs/apple_detection_gt_aligned` |
| `--gt-comparison-dir` | `outputs/apple_detector_comparison_gt_aligned` |
| `--mask-dir` | `outputs/branch_predictions` |
| `--branch-source` | `predicted` or `gt`; default predicted |
| `--reuse-masks` | Read square `<RGB-stem>_branch.png`; skip model loading |
| `--also-gt-control` | Also run centrally aligned GT on the same discovered frames |
| `--threshold` | 0.5; must be within [0,1] |
| `--device` | `auto`, `cpu`, `cuda`; auto uses CUDA when available |
| `--max-distance` | 30 pixels; inclusive centroid tolerance used by comparator |
| `--max-images` | No limit; positive count selects first sorted eligible frames |
| `--skip-comparison` | Only export detection outputs; no reference CSV check |

GT-only control, without branch checkpoint loading:

```powershell
python src/apple_predictor_predicted_compare.py --branch-source gt
```

This writes the **GT** directories, even if `--output-dir` was supplied. Use
`--gt-output-dir`/`--gt-comparison-dir` to override them. GT source plus
`--also-gt-control` is rejected as redundant. For winter use, explicitly set season
and all appropriate modality paths; winter has no assumed GT, and the default
summer RF-DETR reference need not share any winter frames.

## Outputs, matching and limitations

Fresh square masks are saved as `outputs/branch_predictions/<RGB-stem>_branch.png`.
Detection folders contain traditional-schema counts/instances CSVs, annotations,
`image_timings.csv` (including `branch_ms`) and `run_metadata.json`. Comparison
folders contain the comparator's CSVs, manifest, plots and `evaluation_report.md`.
No numerical results are asserted by these instructions. CSVs and matching PNGs
are overwritten; old unmatched PNGs remain. Keep custom output paths distinct
from historical experiments and RF-DETR reference folders.

Counts preserve zero detections. Comparison uses exact basename intersection and
Hungarian one-to-one matching, maximising valid pairs within tolerance before
minimising distance. TP are accepted pairs; FP are unmatched traditional detections;
FN are unmatched reference detections. Precision/recall/F1 return zero for zero
denominators. MAE/RMSE summarise count differences; centroid distance and inclusive
box IoU condition on accepted pairs and are absent when no pairs exist. These are
RF-DETR agreement statistics, not independently verified apple accuracy. The
comparator does not compare runtimes; stage boundaries are described in the output
README. The driver performs no parameter search.

Missing depth is warned/skipped. Requesting GT control also excludes frames with
missing GT from both runs, which can change the evaluated population. Invalid names
are ignored, unreadable images propagate errors, mismatched RGB/depth shapes fail,
and no eligible frames raises RuntimeError. Cached masks with maximum exactly 1 are
rejected; all-zero masks are accepted. Other grayscale values are merely thresholded
>127, rather than validated as strictly 0/255.

**Cache provenance limitation:** reuse does not verify mask fingerprints, checkpoint,
input size or threshold. Metadata records the requested checkpoint/threshold even
when cached masks came from another configuration. Use a fresh mask directory per
checkpoint/threshold and retain independent provenance. A separate executable fix
should validate reuse; this audit changes documentation only.

The wrapper adjusts comparator report paragraphs by exact text prefixes. This is
fragile; updated prose distinguishes standalone defaults from unverified reference
history. Its reproduction paragraph selects the correct branch source but is a
template, not a complete replay of custom arguments. Retain full invocation records
and verify reference history before reporting conclusions.
An aligned GT control reduces the branch-geometry confound but does not remove
model-reference errors, depth-validity differences, correlated frames or leakage.
