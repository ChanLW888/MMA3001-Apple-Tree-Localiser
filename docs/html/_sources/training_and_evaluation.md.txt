# Training and evaluation

## Training configuration

| Setting | Pixel MLP | Basic U-Net | Extended U-Net |
|---|---|---|---|
| Default epochs / batch | 60 / 8 | 60 / 8 | 60 / 8 |
| Loss | BCE with positive-class weight 12 | Plain BCE | Plain BCE |
| Optimiser | Adam | Adam | AdamW, weight decay 1e-4 |
| Default learning rate | 1e-3 | 1e-4 | 3e-4 |
| Early-stop patience | 12 | 12 | 15 |
| LR scheduler | None | None | ReduceLROnPlateau on validation Dice, factor 0.5, patience 4, min 1e-6 |

Defaults are source configuration, not results of an independently verified run. CLI options can override supported values. Supply positive epochs, positive size/batch and a validation fraction strictly between 0 and 1. CLI parsing does not comprehensively validate all numeric ranges; zero epochs can leave history empty and fail during export.

All trainers seed Python/NumPy/PyTorch (default 42), shuffle complete summer samples and reserve `max(1, round(N * val_fraction))` samples for validation (default fraction 0.20). A split without training samples raises RuntimeError. This is a sample-level split, not a held-out-tree protocol, and correlated frames may cross partitions. Seeding does not ensure bitwise deterministic GPU kernels.

Model selection uses highest validation Dice; patience counts epochs without improvement. A nonpositive early-stopping value disables that stop condition. The best state is restored before final validation and panel export. Validation is reused for model selection, so final metrics are not independent test performance.

## Metrics

Prediction is sigmoid(logit) >=threshold (default 0.5); target foreground is >=0.5. TP/FP/FN are summed over all validation pixels. Dice is `2TP/(2TP+FP+FN+eps)`, IoU `TP/(TP+FP+FN+eps)`, precision `TP/(TP+FP+eps)` and recall `TP/(TP+FN+eps)`, with eps=1e-8. Empty foreground therefore scores zero. Batch loss is weighted by sample count and divided by dataset length, rather than averaged equally per batch.

`history.csv` stores epoch/training loss/validation metrics. `val_metrics.csv` stores final best-state validation metrics. Checkpoints and comparison panels are described in the output README.

## Winter and apple interpretation

Winter inference produces qualitative branch masks and synchronised model-forward timing; it requires no GT and cannot report winter segmentation accuracy. Timing excludes preprocessing, transfer, sigmoid/thresholding and saving. Hardware/runtime results are not claimed here.

Apple comparison is a separate workflow on saved CSVs. Its RF-DETR reference is pseudo-ground truth, and branch GT masks are not apple annotations. Reference-relative F1, count agreement and conditional accepted-match distances must not be presented as independently verified detection/localisation accuracy. See the root README for different branch alignment rules and other confounders.

## Testing boundaries

Synthetic matching tests can validate threshold inclusivity, one-to-one cardinality, metrics and CSV validation without orchard inference. Full test discovery also includes an all-frame detector regression and output-dependent fixtures. A stale CSV-path mock, missing export timing fields and historical output-directory assumptions are documented in the verification report; they were not fixed in this documentation-only task.
