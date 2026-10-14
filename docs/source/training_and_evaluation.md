# Training and evaluation

## Training procedure

The documented branch-segmentation scripts use a shuffled train/validation
split. Training samples are shuffled with a configured random seed, and model
selection is based on validation Dice.

The scripts use binary cross-entropy with logits (BCEWithLogitsLoss). The
improved model uses AdamW and a validation-Dice-driven learning-rate scheduler;
the MLP and basic U-Net use their own configured optimiser settings. Refer to
the actual script and command-line arguments for the exact values used in a
given run.

Early stopping can terminate training when validation Dice does not improve
for the configured patience. The best checkpoint is saved according to
validation Dice.

## Metrics

For binary branch segmentation:

- **Dice:** overlap between predicted and ground-truth foreground masks.
- **IoU:** intersection divided by union of the foreground masks.
- **Precision:** fraction of predicted foreground pixels that are correct.
- **Recall:** fraction of ground-truth foreground pixels recovered.

The probability threshold controls conversion from model logits to a binary
prediction and affects the reported overlap metrics.

## Interpretation limits

The validation set is used for model selection and is not an independent test
set. Validation results should not be described as unbiased test performance.
Winter predictions without ground-truth masks support qualitative inspection
only, not quantitative claims about winter generalisation.
