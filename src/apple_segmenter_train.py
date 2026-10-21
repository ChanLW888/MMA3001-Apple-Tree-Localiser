"""Train the RF-DETR medium apple instance-segmentation model.

Uses cleaned COCO train/valid splits in data/apple_tree_clean. Requests 50 epochs and
batch size 1; remaining defaults belong to rfdetr. Writes outputs/rfdetr_apple_tree.
Import checks paths and prints configuration; training requires main. Cached pretrained
weights may be needed offline.

Raises:
    FileNotFoundError: If data folders or COCO files are missing.
    ImportError: If rfdetr is unavailable.

Example:
    python src/apple_segmenter_train.py
"""

from pathlib import Path

from rfdetr import RFDETRSegMedium


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# CLEANED APPLE-ONLY DATASET
DATASET_DIR = (
    ROOT
    / "data"
    / "apple_tree_clean"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "rfdetr_apple_tree"
)


# ============================================================
# CHECK DATASET
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

TRAIN_ANNOTATIONS = TRAIN_DIR / "_annotations.coco.json"
VALID_ANNOTATIONS = VALID_DIR / "_annotations.coco.json"


if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR}"
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Train folder not found:\n{TRAIN_DIR}"
    )

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VALID_DIR}"
    )

if not TRAIN_ANNOTATIONS.exists():
    raise FileNotFoundError(
        f"Training annotations not found:\n{TRAIN_ANNOTATIONS}"
    )

if not VALID_ANNOTATIONS.exists():
    raise FileNotFoundError(
        f"Validation annotations not found:\n{VALID_ANNOTATIONS}"
    )


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 60)
print("RF-DETR APPLE INSTANCE SEGMENTATION")
print("=" * 60)

print(f"Dataset : {DATASET_DIR}")
print(f"Train   : {TRAIN_DIR}")
print(f"Valid   : {VALID_DIR}")
print(f"Output  : {OUTPUT_DIR}")

print("=" * 60)


# ============================================================
# TRAIN
# ============================================================

def main():

    """Train RFDETRSegMedium on cleaned apple COCO splits.

    Uses 50 epochs, batch size 1 and library defaults otherwise. Writes
    outputs/rfdetr_apple_tree. Module checks dataset paths at import. Construction may
    retrieve pretrained weights if uncached.

    Returns:
        None.

    Raises:
        OSError: If data, pretrained weights or output cannot be accessed.
        RuntimeError: If training fails.
    """
    print("\nLoading RF-DETR Segmentation Medium...")

    model = RFDETRSegMedium()

    print("Model loaded.")

    print("\nStarting training...")
    print("Classes: apple")
    print("Epochs : 50")
    print("Batch  : 1")
    print()

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=50,
        batch_size=1,
        output_dir=str(OUTPUT_DIR),
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("Output directory:")
    print(OUTPUT_DIR)

    print("=" * 60)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()