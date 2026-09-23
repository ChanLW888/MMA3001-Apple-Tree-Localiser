from pathlib import Path
import csv

import cv2
import numpy as np
import torch

from rfdetr import RFDETRSegMedium


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# TreeID RGB images
# ------------------------------------------------------------

RGB_DIR = (
    ROOT
    / "data"
    / "Organised"
    / "Summer"
    / "RGB"
)


# ------------------------------------------------------------
# TRAINED RF-DETR MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    ROOT
    / "outputs"
    / "rfdetr_apple_tree"
    / "checkpoint_best_total.pth"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "apple_rfdetr"
)

ANNOTATED_DIR = OUTPUT_DIR / "annotated"
MASK_DIR = OUTPUT_DIR / "masks"


# ============================================================
# INFERENCE SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.20


# ============================================================
# PROCESSING
# ============================================================

# False = process the entire dataset
TEST_ONLY = False

# Only used when TEST_ONLY = True
TEST_FRAMES = 5


# ============================================================
# SETUP
# ============================================================

ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 60)
print("RF-DETR Apple Instance Segmentation")
print("=" * 60)

print(f"RGB directory : {RGB_DIR}")
print(f"Model         : {MODEL_PATH}")
print(f"Output        : {OUTPUT_DIR}")
print(f"Confidence    : {CONFIDENCE_THRESHOLD}")
print(f"Device        : {device}")

if torch.cuda.is_available():

    print(
        f"GPU           : "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    print(
        "WARNING: CUDA not available. "
        "Running on CPU."
    )

print("=" * 60)


# ============================================================
# CHECK PATHS
# ============================================================

if not RGB_DIR.exists():

    raise FileNotFoundError(
        f"RGB directory not found:\n{RGB_DIR}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"RF-DETR checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained RF-DETR model...")

model = RFDETRSegMedium(
    pretrain_weights=str(MODEL_PATH)
)

print("RF-DETR model loaded.")


# ============================================================
# GET IMAGES
# ============================================================

image_paths = sorted(
    RGB_DIR.glob("*.png")
)

if len(image_paths) == 0:

    raise RuntimeError(
        f"No PNG images found in:\n{RGB_DIR}"
    )


if TEST_ONLY:

    image_paths = image_paths[:TEST_FRAMES]


print(
    f"\nProcessing "
    f"{len(image_paths)} image(s)...\n"
)


# ============================================================
# CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "apple_counts.csv"
)


csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)


csv_writer = csv.writer(csv_file)


csv_writer.writerow([
    "image",
    "apple_count"
])


# ============================================================
# PROCESS IMAGES
# ============================================================

total_apples = 0


for image_index, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"[{image_index}/{len(image_paths)}] "
        f"{image_path.name}"
    )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "  WARNING: Could not read image."
        )

        continue


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # RF-DETR inference
    # --------------------------------------------------------

    detections = model.predict(
        str(image_path),
        threshold=CONFIDENCE_THRESHOLD
    )


    # --------------------------------------------------------
    # Prepare outputs
    # --------------------------------------------------------

    annotated = image.copy()

    instance_mask = np.zeros(
        (height, width),
        dtype=np.uint16
    )

    apple_count = 0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if detections is not None:

        if (
            detections.mask is not None
            and len(detections) > 0
        ):

            masks = detections.mask

            confidences = detections.confidence


            # ------------------------------------------------
            # Process each detected apple
            # ------------------------------------------------

            for detection_index in range(
                len(detections)
            ):

                confidence = float(
                    confidences[detection_index]
                )


                if confidence < CONFIDENCE_THRESHOLD:

                    continue


                # ------------------------------------------------
                # Get segmentation mask
                # ------------------------------------------------

                binary_mask = (
                    masks[detection_index] > 0.5
                )


                # ------------------------------------------------
                # Resize mask if necessary
                # ------------------------------------------------

                if binary_mask.shape != (
                    height,
                    width
                ):

                    binary_mask = cv2.resize(
                        binary_mask.astype(
                            np.uint8
                        ),
                        (
                            width,
                            height
                        ),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)


                # ------------------------------------------------
                # Check mask area
                # ------------------------------------------------

                area = int(
                    binary_mask.sum()
                )


                if area == 0:

                    continue


                # ------------------------------------------------
                # Count apple
                # ------------------------------------------------

                apple_count += 1

                apple_id = apple_count


                # ------------------------------------------------
                # Store instance ID
                #
                # 0 = background
                # 1 = apple 1
                # 2 = apple 2
                # etc.
                # ------------------------------------------------

                instance_mask[
                    binary_mask
                ] = apple_id


    # ========================================================
    # CREATE RED MASK OVERLAY
    # ========================================================

    if apple_count > 0:

        # Create red image
        red_overlay = np.zeros_like(
            annotated
        )

        red_overlay[:, :] = (
            0,
            0,
            255
        )


        # Combined mask containing all apples
        combined_mask = (
            instance_mask > 0
        )


        # Apply red overlay only where apples are detected
        annotated[combined_mask] = cv2.addWeighted(
            annotated[combined_mask],
            0.65,
            red_overlay[combined_mask],
            0.35,
            0
        )


    # ========================================================
    # DRAW TOTAL COUNT
    # ========================================================

    count_text = (
        f"Apples: {apple_count}"
    )


    cv2.putText(
        annotated,
        count_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )


    # ========================================================
    # SAVE ANNOTATED IMAGE
    # ========================================================

    output_name = image_path.stem


    annotated_path = (
        ANNOTATED_DIR
        / f"{output_name}_annotated.png"
    )


    cv2.imwrite(
        str(annotated_path),
        annotated
    )


    # ========================================================
    # SAVE INSTANCE MASK
    # ========================================================

    mask_path = (
        MASK_DIR
        / f"{output_name}_instances.png"
    )


    cv2.imwrite(
        str(mask_path),
        instance_mask
    )


    # ========================================================
    # SAVE COUNT TO CSV
    # ========================================================

    csv_writer.writerow([
        image_path.name,
        apple_count
    ])


    # ========================================================
    # TOTAL
    # ========================================================

    total_apples += apple_count


    print(
        f"  Apples detected: "
        f"{apple_count}"
    )


# ============================================================
# FINISH
# ============================================================

csv_file.close()


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Images processed : "
    f"{len(image_paths)}"
)

print(
    f"Total detections : "
    f"{total_apples}"
)

print(
    f"CSV              : "
    f"{csv_path}"
)

print(
    f"Annotated images : "
    f"{ANNOTATED_DIR}"
)

print(
    f"Instance masks   : "
    f"{MASK_DIR}"
)

print("=" * 60)