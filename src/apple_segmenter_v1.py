"""Run branch-guided RF-DETR apple instance segmentation.

Reads outputs/rfdetr_apple_tree/checkpoint_best_total.pth and writes
outputs/apple_rfdetr_instances. Defaults to summer branch GT (>127) with central
placement. Predicted mode only reads exported <RGB-stem>_branch.png masks. Does not run
branch inference or consume depth. Model detections are not verified apple annotations.

Example:
    python src/apple_segmenter_v1.py
"""
from pathlib import Path
import csv
import re
import time

import torch

import cv2
import numpy as np

# ------------------------- CONFIGURATION -------------------------
ROOT = Path(__file__).resolve().parents[1]
SEASON = "Summer"
DATA_DIR = ROOT / "data" / "Organised" / SEASON
RGB_DIR = DATA_DIR / "RGB"
GT_DIR = DATA_DIR / "GT"
PREDICTED_BRANCH_DIR = ROOT / "outputs" / "branch_predictions"

RFDETR_CHECKPOINT = (
    ROOT / "outputs" / "rfdetr_apple_tree" / "checkpoint_best_total.pth"
)
OUTPUT_DIR = ROOT / "outputs" / "apple_rfdetr_instances"

# "gt" uses the original branch GT masks, like your traditional detector.
# "predicted" loads already-exported branch masks from PREDICTED_BRANCH_DIR.
BRANCH_SOURCE = "gt"
CONFIDENCE_THRESHOLD = 0.20
INSTANCE_MASK_THRESHOLD = 0.50
BRANCH_DILATION = 100  # ellipse kernel DIAMETER in source-image pixels
MIN_BRANCH_OVERLAP = 0.05
KEEP_LARGEST_BRANCH_COMPONENT = False
MIN_APPLE_AREA = 1
TEST_ONLY = False
TEST_FRAMES = 5
# Set START_FRAME = 84 to continue from frame 84 after a previous crash.
# Set it back to 1 for a complete fresh run. CSVs are recreated on each run.
START_FRAME = 1

# Assumption: GT describes the square centre crop used by branch preprocessing.
# For 640x480 RGB this is the central 480x480 field of view; verify annotation
# provenance separately. "full_resize" instead stretches GT over the full RGB
# grid, as the traditional detector does.
BRANCH_MASK_ALIGNMENT = "center_crop"
# -----------------------------------------------------------------


def parse_rgb_filename(path: Path):
    """Parse tree, season and time from an RGB PNG basename.

    Args:
        path: pathlib.Path; expected
            R<digits>N<digits>_<Summer|Winter>_RGB-<time>.png.

    Returns:
        dict[str, str] or None: tree, season and time in original case, or None for
            unmatched names.
    """
    match = re.fullmatch(
        r"(?P<tree>R\d+N\d+)_(?P<season>Summer|Winter)_RGB-"
        r"(?P<time>.+)\.png",
        path.name,
        flags=re.IGNORECASE,
    )
    return match.groupdict() if match else None


def corresponding_branch_path(image_path: Path, info: dict) -> Path:
    """Resolve a GT or externally exported predicted branch mask.

    Default source is GT. This does not run a branch network or verify predicted-mask
    provenance.

    Args:
        image_path: pathlib.Path to the RGB frame.
        info: dict[str, str]; tree, season and time.

    Returns:
        pathlib.Path: GT path or <RGB-stem>_branch.png in PREDICTED_BRANCH_DIR.

    Raises:
        ValueError: If BRANCH_SOURCE is not gt or predicted.
        KeyError: If GT metadata is incomplete.
    """
    if BRANCH_SOURCE == "gt":
        return GT_DIR / (
            f'{info["tree"]}_{info["season"]}_GT-{info["time"]}.png'
        )
    if BRANCH_SOURCE == "predicted":
        # Adjust this naming convention if your exported U-Net masks differ.
        return PREDICTED_BRANCH_DIR / (image_path.stem + "_branch.png")
    raise ValueError("BRANCH_SOURCE must be 'gt' or 'predicted'")


def align_branch_mask(mask: np.ndarray, image_shape, alignment: str):
    """Place a binary branch mask in original RGB coordinates.

    Thresholds >127 before resizing. Same-size masks return directly. center_crop
    expands to min(H, W) square and centres on a zero canvas; full_resize stretches over
    the frame. Neither verifies true GT geometry. Input encoded as 0/1 becomes empty.

    Args:
        mask: numpy.ndarray; grayscale [Hm, Wm] or BGR mask, expected 0/255
            encoding.
        image_shape: tuple[int, ...]; starts with original (height, width).
        alignment: str; full_resize or center_crop when shapes differ.

    Returns:
        numpy.ndarray: uint8 [H, W] branch mask in {0, 1}.

    Raises:
        ValueError: If shapes differ and alignment is unsupported.
        cv2.error: If conversion or resizing fails.
    """
    height, width = image_shape[:2]
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 127).astype(np.uint8)  # binary 0/1

    if mask.shape == (height, width):
        return mask

    if alignment == "full_resize":
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if alignment != "center_crop":
        raise ValueError("Unknown BRANCH_MASK_ALIGNMENT value")

    crop_size = min(height, width)
    # Match the branch network's square, centre-cropped RGB input.
    square = cv2.resize(
        mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((height, width), dtype=np.uint8)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    canvas[y0:y0 + crop_size, x0:x0 + crop_size] = square
    return canvas


def make_branch_zone(branch_mask: np.ndarray):
    """Optionally retain the largest branch component, then dilate.

    BRANCH_DILATION is ellipse diameter in source-image pixels. Largest-component
    filtering is disabled by default.

    Args:
        branch_mask: numpy.ndarray; aligned [H, W], foreground >0.

    Returns:
        numpy.ndarray: uint8 [H, W] proximity zone in {0, 1}.

    Raises:
        ValueError: If BRANCH_DILATION <1.
        cv2.error: If component analysis or morphology fails.
    """
    binary = (branch_mask > 0).astype(np.uint8)
    if KEEP_LARGEST_BRANCH_COMPONENT and np.any(binary):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = (labels == component).astype(np.uint8)

    if BRANCH_DILATION < 1:
        raise ValueError("BRANCH_DILATION must be >= 1")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BRANCH_DILATION, BRANCH_DILATION)
    )
    return cv2.dilate(binary, kernel)


def filter_instances(detections, image_shape, zone):
    """Retain whole RF-DETR masks with sufficient branch-zone overlap.

    Missing masks/confidences yield no instances. Current constants require confidence
    >=0.20, mask >0.50, area >=1 and overlap >=0.05. Uses nearest-neighbour resizing. No
    class-ID filtering occurs; checkpoint is assumed apple-only. Masks remain whole, not
    clipped to the zone.

    Args:
        detections: RF-DETR/supervision detections with mask [N, Hm, Wm] and
            confidence [N], or None.
        image_shape: tuple[int, ...]; original (H, W, ...) shape.
        zone: numpy.ndarray; aligned binary [H, W] proximity region.

    Returns:
        list[dict]: mask (bool [H, W]), confidence, area (pixels), overlap,
            inclusive xyxy bbox and mean (x, y) centroid.

    Raises:
        ValueError: If a mask is not 2D or zone cannot broadcast.
        cv2.error: If resizing fails.
    """
    height, width = image_shape[:2]
    kept = []
    if detections is None or len(detections) == 0:
        return kept
    masks = detections.mask
    confidences = detections.confidence
    if masks is None or confidences is None:
        return kept

    for idx in range(len(detections)):
        confidence = float(confidences[idx])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        raw_mask = np.asarray(masks[idx])
        if raw_mask.ndim != 2:
            raise ValueError(f"RF-DETR mask {idx} has shape {raw_mask.shape}")
        # Threshold before nearest-neighbour resizing to keep a discrete instance mask.
        binary = (raw_mask > INSTANCE_MASK_THRESHOLD).astype(np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(
                binary, (width, height), interpolation=cv2.INTER_NEAREST
            )
        apple_area = int(np.count_nonzero(binary))
        if apple_area < MIN_APPLE_AREA:
            continue

        intersection = int(np.count_nonzero((binary > 0) & (zone > 0)))
        overlap = intersection / apple_area
        if overlap < MIN_BRANCH_OVERLAP:
            continue

        ys, xs = np.nonzero(binary)
        kept.append({
            "mask": binary.astype(bool),
            "confidence": confidence,
            "area": apple_area,
            "overlap": overlap,
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()),
                     int(ys.max())),
            "centroid": (float(xs.mean()), float(ys.mean())),
        })
    return kept


def render_apple_annotations(image_bgr, boxes, centroids, color_bgr):
    """Render boxes, centres, indices and total count on a copy.

    zip truncates unequal box/centroid sequences; callers should supply one centre per
    box.

    Args:
        image_bgr: numpy.ndarray; uint8 [H, W, 3] OpenCV BGR.
        boxes: Sequence of inclusive (x1, y1, x2, y2) boxes in pixels.
        centroids: Sequence of floating (x, y) centres paired with boxes.
        color_bgr: tuple[int, int, int]; BGR colour in [0, 255].

    Returns:
        numpy.ndarray: Annotated BGR copy with original shape/dtype.

    Raises:
        cv2.error: If drawing arguments are invalid.
    """
    annotated = image_bgr.copy()
    for index, (box, center) in enumerate(zip(boxes, centroids), start=1):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (int(round(float(v))) for v in center)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.circle(annotated, (cx, cy), 3, color_bgr, -1, cv2.LINE_AA)
        cv2.putText(annotated, str(index), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
    cv2.putText(annotated, f'Apples: {len(boxes)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
    return annotated


def save_outputs(image, image_path, branch, zone, instances):
    """Write cyan annotations, uint16 instance IDs and branch-zone PNGs.

    ID 0 is background; instances start at 1. Later masks overwrite earlier IDs at
    overlaps; CSV areas still describe whole instances.

    Args:
        image: numpy.ndarray; uint8 [H, W, 3] BGR.
        image_path: pathlib.Path; stem names saved files.
        branch: numpy.ndarray; accepted but unused.
        zone: numpy.ndarray; [H, W] binary 0/1 proximity mask.
        instances: list[dict] from filter_instances.

    Returns:
        tuple[pathlib.Path, pathlib.Path, pathlib.Path]: Annotated image,
            instance-ID mask and zone image.

    Raises:
        ValueError: If count exceeds uint16 capacity.
        OSError: If directories or image files cannot be written.
        cv2.error: If encoding or drawing fails.
    """
    out_ann = OUTPUT_DIR / "annotated"
    out_masks = OUTPUT_DIR / "masks"
    out_zones = OUTPUT_DIR / "branch_zones"
    for folder in (out_ann, out_masks, out_zones):
        folder.mkdir(parents=True, exist_ok=True)

    h, w = image.shape[:2]
    if len(instances) > np.iinfo(np.uint16).max:
        raise ValueError("Too many instances for uint16 ID mask")
    id_map = np.zeros((h, w), dtype=np.uint16)
    # Segmentation masks are saved separately; no coloured mask overlay.
    for apple_id, instance in enumerate(instances, start=1):
        # Later IDs win overlap pixels; CSV geometry still describes whole masks.
        id_map[instance["mask"]] = apple_id

    boxes = [instance["bbox"] for instance in instances]
    centers = [instance["centroid"] for instance in instances]
    annotated = render_apple_annotations(image, boxes, centers, (255, 255, 0))

    stem = image_path.stem
    paths = (
        out_ann / f"{stem}_annotated.png",
        out_masks / f"{stem}_instances.png",
        out_zones / f"{stem}_zone.png",
    )
    for path, array in zip(paths, (
        annotated, id_map, (zone * 255).astype(np.uint8)
    )):
        if not cv2.imwrite(str(path), array):
            raise IOError(f"Could not write {path}")
    return paths


def main():
    """Load RF-DETR and export branch-guided apple predictions.

    Defaults to summer branch GT with central placement. Skips missing/unreadable inputs
    and unexpected names. Recreates CSVs even for START_FRAME >1. model.predict reads
    RGB paths; no depth input is used. Initialisation may require cached auxiliary
    weights.

    Returns:
        None.

    Raises:
        FileNotFoundError: If RGB directory or checkpoint is missing.
        ImportError: If rfdetr cannot be imported.
        ValueError: If overlap/start-frame settings, mask rank or alignment are
            invalid.
        RuntimeError: If no selected PNGs exist or model execution fails.
        OSError: If outputs cannot be written.
    """
    if not RGB_DIR.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {RGB_DIR}")
    if not RFDETR_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing RF-DETR checkpoint: {RFDETR_CHECKPOINT}")
    if not 0 <= MIN_BRANCH_OVERLAP <= 1:
        raise ValueError("MIN_BRANCH_OVERLAP must be between 0 and 1")

    try:
        from rfdetr import RFDETRSegMedium
    except ImportError as exc:
        raise ImportError("Install the rfdetr package in your Conda environment") from exc

    print(f"RUNNING SCRIPT: {Path(__file__).resolve()}")
    print(f"ANNOTATION MODE: CYAN BOXES + CYAN CENTROIDS (NO OVERLAY)")
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")
    model = RFDETRSegMedium(pretrain_weights=str(RFDETR_CHECKPOINT))
    images = sorted(RGB_DIR.glob("*.png"))
    if TEST_ONLY:
        images = images[:TEST_FRAMES]
    if START_FRAME < 1:
        raise ValueError("START_FRAME must be >= 1")
    if START_FRAME > 1:
        images = images[START_FRAME - 1:]
    if not images:
        raise RuntimeError(f"No PNG images in {RGB_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts_path = OUTPUT_DIR / "apple_counts.csv"
    details_path = OUTPUT_DIR / "apple_instances.csv"
    processed = skipped = 0
    measured_inference_ms = []
    measured_total_ms = []
    timing_path = OUTPUT_DIR / "image_timings.csv"

    def sync_cuda():
        # CUDA kernels execute asynchronously; sync for accurate GPU inference timing.
        """Synchronise pending CUDA work for inference timing.

        Does nothing without CUDA; barriers keep asynchronous kernels inside the timed
        interval.

        Returns:
            None.

        Raises:
            RuntimeError: If CUDA synchronisation fails.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with counts_path.open("w", newline="", encoding="utf-8") as cf, \
         details_path.open("w", newline="", encoding="utf-8") as df, \
         timing_path.open("w", newline="", encoding="utf-8") as tf:
        counts = csv.writer(cf)
        details = csv.writer(df)
        timing = csv.writer(tf)
        timing.writerow(["image", "rf_detr_detections", "kept_apple_count",
                         "read_ms", "inference_ms", "filter_ms",
                         "save_ms", "total_ms"])
        counts.writerow([
            "image", "tree", "season", "time",
            "branch_source", "rf_detr_detections", "kept_apple_count"
        ])
        details.writerow([
            "image", "apple_id", "confidence", "area_px",
            "branch_overlap", "cx_px", "cy_px",
            "x1", "y1", "x2", "y2"
        ])

        for i, image_path in enumerate(images, 1):
            info = parse_rgb_filename(image_path)
            if info is None:
                print(f"[{i}] Skipped unexpected filename: {image_path.name}")
                skipped += 1
                continue

            branch_path = corresponding_branch_path(image_path, info)
            if not branch_path.is_file():
                print(f"[{i}] Skipped missing branch mask: {branch_path.name}")
                skipped += 1
                continue
            frame_start = time.perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            raw_branch = cv2.imread(str(branch_path), cv2.IMREAD_UNCHANGED)
            if image is None or raw_branch is None:
                print(f"[{i}] Skipped unreadable image/mask: {image_path.name}")
                skipped += 1
                continue

            branch = align_branch_mask(
                raw_branch, image.shape, BRANCH_MASK_ALIGNMENT
            )
            zone = make_branch_zone(branch)
            read_ms = (time.perf_counter() - frame_start) * 1000
            sync_cuda()
            infer_start = time.perf_counter()
            predictions = model.predict(
                str(image_path), threshold=CONFIDENCE_THRESHOLD
            )
            sync_cuda()
            inference_ms = (time.perf_counter() - infer_start) * 1000
            filter_start = time.perf_counter()
            kept = filter_instances(predictions, image.shape, zone)
            raw_count = len(predictions) if predictions is not None else 0
            filter_ms = (time.perf_counter() - filter_start) * 1000
            save_start = time.perf_counter()
            save_outputs(image, image_path, branch, zone, kept)
            counts.writerow([
                image_path.name, info["tree"], info["season"], info["time"],
                BRANCH_SOURCE, raw_count, len(kept)
            ])
            for j, apple in enumerate(kept, 1):
                details.writerow([
                    image_path.name, j, f'{apple["confidence"]:.5f}',
                    apple["area"], f'{apple["overlap"]:.5f}',
                    f'{apple["centroid"][0]:.2f}',
                    f'{apple["centroid"][1]:.2f}', *apple["bbox"]
                ])
            save_ms = (time.perf_counter() - save_start) * 1000
            total_ms = (time.perf_counter() - frame_start) * 1000
            timing.writerow([image_path.name, raw_count, len(kept),
                            f"{read_ms:.4f}", f"{inference_ms:.4f}",
                            f"{filter_ms:.4f}", f"{save_ms:.4f}", f"{total_ms:.4f}"])
            measured_inference_ms.append(inference_ms)
            measured_total_ms.append(total_ms)
            processed += 1
            print(f"[{i}/{len(images)}] {image_path.name}: "
                  f"RF-DETR={raw_count}, along branch={len(kept)} | "
                  f"inference={inference_ms:.2f} ms | total={total_ms:.2f} ms")

    print(f"\nDone: processed={processed}, skipped={skipped}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Timings: {timing_path}")
    if measured_inference_ms:
        print(f"Mean RF-DETR inference: {np.mean(measured_inference_ms):.2f} ms/image")
        print(f"Mean full-frame runtime: {np.mean(measured_total_ms):.2f} ms/image")


if __name__ == "__main__":
    main()
