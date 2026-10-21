"""Detect red apples with depth validity and branch ground truth.

A 2D image-space baseline, not metric 3D localisation. Branch GT (>0) is stretched to
the full RGB grid and dilated. RGB/depth must be aligned. Writes annotation PNGs,
counts, instances, timings and metadata under outputs/apple_detection.

Example:
    python src/apple_predictor_v1.py
"""
from pathlib import Path
import csv
import json
import re
import warnings
import time

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / 'data' / 'Organised'
SEASON = 'Summer'
RGB_DIR = DATASET / SEASON / 'RGB'
DEPTH_DIR = DATASET / SEASON / 'D'
GT_DIR = DATASET / SEASON / 'GT'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'apple_detection'
ANNOTATED_DIR = OUTPUT_ROOT  # Save all annotated images in one flat folder
COUNTS_CSV = OUTPUT_ROOT / 'apple_counts.csv'
INSTANCES_CSV = OUTPUT_ROOT / 'apple_instances.csv'
TIMINGS_CSV = OUTPUT_ROOT / 'image_timings.csv'

RED_THRESHOLD = 50
RED_DOMINANCE = 1.5
MIN_APPLE_AREA = 30
MAX_APPLE_AREA = 5000
MIN_CIRCULARITY = 0.20
MAX_ASPECT_RATIO = 2.5
BRANCH_DILATION = 100
MIN_BRANCH_OVERLAP = 0.05
MORPH_KERNEL_SIZE = 5
# Preserve original script's GT-to-image alignment (full-frame resize).
# Confirm this agrees with your dataset before interpreting localisation metrics.

INSTANCE_FIELDS = [
    'image', 'apple_id', 'confidence', 'area_px', 'branch_overlap',
    'cx_px', 'cy_px', 'x1', 'y1', 'x2', 'y2',
    'red_score', 'image_width', 'image_height',
]


def parse_rgb_filename(path):
    """Parse tree, season and time from an RGB PNG basename.

    Args:
        path: pathlib.Path; expected
            R<digits>N<digits>_<Summer|Winter>_RGB-<time>.png.

    Returns:
        dict[str, str] or None: tree, season and time in original case, or None for
            unmatched names.
    """
    match = re.fullmatch(
        r'(?P<tree>R\d+N\d+)_(?P<season>Summer|Winter)_RGB-(?P<time>.+)\.png',
        path.name,
        re.IGNORECASE,
    )
    return match.groupdict() if match else None


def load_rgb(path):
    """Read an image in RGB channel order.

    Args:
        path: str or pathlib.Path; readable image path.

    Returns:
        numpy.ndarray: uint8 [H, W, 3] RGB in [0, 255].

    Raises:
        OSError: If image opening or decoding fails.
    """
    return np.asarray(Image.open(path).convert('RGB'))


def load_depth(path):
    """Read stored depth intensities without metric conversion.

    Args:
        path: str or pathlib.Path; readable depth image.

    Returns:
        numpy.ndarray: [H, W] in stored dtype; first channel is used for
            multichannel images.

    Raises:
        OSError: If image opening or decoding fails.
    """
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def load_gt(path, target_shape):
    """Load branch ground truth and stretch it to the RGB grid.

    Nearest-neighbour interpolation preserves labels. Full-frame stretching differs from
    RF-DETR default central placement; it does not verify annotation geometry.

    Args:
        path: str or pathlib.Path; branch GT image, not apple-instance annotation.
        target_shape: tuple[int, int]; positive original (height, width) in pixels.

    Returns:
        numpy.ndarray: uint8 [H, W] grayscale GT; foreground is interpreted
            downstream as >0.

    Raises:
        OSError: If GT cannot be decoded.
        cv2.error: If resize dimensions are invalid.
    """
    gt = np.asarray(Image.open(path).convert('L'))
    if gt.shape != target_shape:
        gt = cv2.resize(
            gt, (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return gt


def detect_red_mask(rgb, depth):
    """Select red-dominant pixels with valid depth.

    Floating channel arithmetic avoids uint8 overflow. Thresholds are module constants;
    depth supplies validity, not 3D coordinates.

    Args:
        rgb: numpy.ndarray; [H, W, 3] RGB intensities in [0, 255].
        depth: numpy.ndarray; aligned [H, W] values, valid when >0.

    Returns:
        numpy.ndarray: uint8 [H, W] binary mask.

    Raises:
        ValueError: If depth cannot broadcast to the RGB grid.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return ((r > RED_THRESHOLD) & (r > g * RED_DOMINANCE)
            & (r > b * RED_DOMINANCE) & (depth > 0)).astype(np.uint8)


def clean_red_mask(mask):
    """Suppress isolated red noise and close small gaps.

    The ellipse is MORPH_KERNEL_SIZE pixels wide/high. Morphology may merge nearby
    regions; components are candidates rather than guaranteed apples.

    Args:
        mask: numpy.ndarray; uint8 [H, W] binary red mask.

    Returns:
        numpy.ndarray: uint8 [H, W] opened then closed mask.

    Raises:
        cv2.error: If mask or kernel is invalid.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    # Opening removes isolated red noise; closing reconnects small within-region gaps.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def create_branch_zone(gt):
    """Dilate labelled branch pixels into a proximity region.

    BRANCH_DILATION is ellipse diameter in source-image pixels, not radius. Uses
    ground-truth branches, not predicted segmentation.

    Args:
        gt: numpy.ndarray; aligned [H, W] branch GT, foreground >0.

    Returns:
        numpy.ndarray: uint8 [H, W] zone in {0, 1}.

    Raises:
        cv2.error: If morphology input or kernel is invalid.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BRANCH_DILATION, BRANCH_DILATION)
    )
    return cv2.dilate((gt > 0).astype(np.uint8), kernel)


def find_apple_candidates(rgb, depth, gt):
    """Filter connected red components by geometry and branch proximity.

    Original-image coordinates: top-left origin, x right, y down. Overlap is
    intersection/whole-component area. Red score is mean R/(G+B+1), not calibrated
    confidence. No apple labels are read.

    Args:
        rgb: numpy.ndarray; uint8 [H, W, 3] RGB.
        depth: numpy.ndarray; aligned [H, W] depth-validity intensities.
        gt: numpy.ndarray; aligned [H, W] branch ground truth.

    Returns:
        list[dict]: label, area (pixels), bbox (x, y, width, height), centroid (x,
            y), circularity, aspect_ratio, branch_overlap and red_score.

    Raises:
        ValueError: If RGB/depth/GT spatial shapes differ.
        cv2.error: If image processing fails.
    """
    if rgb.shape[:2] != depth.shape or depth.shape != gt.shape:
        raise ValueError('RGB, depth and branch GT masks must be aligned')
    red_mask = clean_red_mask(detect_red_mask(rgb, depth))
    branch_zone = create_branch_zone(gt)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8
    )
    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (MIN_APPLE_AREA <= area <= MAX_APPLE_AREA):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        # Degenerate contours have no meaningful circularity and would divide by zero.
        if perimeter == 0:
            continue
        # Preserve the original filtering rule for a comparable baseline.
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        local_branch = branch_zone[y:y + h, x:x + w]
        overlap_pixels = np.count_nonzero((component > 0) & (local_branch > 0))
        # Use the whole red component as denominator, so proximity is a fraction.
        branch_overlap = overlap_pixels / area
        if branch_overlap < MIN_BRANCH_OVERLAP:
            continue
        region = rgb[y:y + h, x:x + w][component > 0].astype(np.float32)
        red_score = float(np.mean(region[:, 0] / (region[:, 1] + region[:, 2] + 1)))
        candidates.append({
            'label': label, 'area': area, 'bbox': (x, y, w, h),
            'centroid': (float(centroids[label, 0]), float(centroids[label, 1])),
            'circularity': float(circularity), 'aspect_ratio': float(aspect_ratio),
            'branch_overlap': float(branch_overlap), 'red_score': red_score,
        })
    return candidates


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


def draw_annotations(rgb, candidates):
    """Convert candidate boxes to inclusive corners and draw orange labels.

    Args:
        rgb: numpy.ndarray; uint8 [H, W, 3] RGB.
        candidates: list[dict] from find_apple_candidates.

    Returns:
        numpy.ndarray: Annotated RGB copy for Pillow saving.

    Raises:
        KeyError: If bbox or centroid is missing.
        cv2.error: If colour conversion or drawing fails.
    """
    boxes = []
    centers = []
    for apple in candidates:
        x, y, w, h = apple['bbox']
        boxes.append((x, y, x + w - 1, y + h - 1))
        centers.append(apple['centroid'])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rendered = render_apple_annotations(bgr, boxes, centers, (0, 165, 255))
    return cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)


def get_matching_files(rgb_path, info):
    """Build depth and branch-GT paths for a parsed frame.

    Args:
        rgb_path: pathlib.Path; accepted but unused.
        info: dict[str, str]; tree, season and time from parse_rgb_filename.

    Returns:
        tuple[pathlib.Path, pathlib.Path]: Depth and GT paths; existence is not
            checked.

    Raises:
        KeyError: If required metadata is missing.
    """
    depth_path = DEPTH_DIR / f'{info["tree"]}_{info["season"]}_D-{info["time"]}.png'
    gt_path = GT_DIR / f'{info["tree"]}_{info["season"]}_GT-{info["time"]}.png'
    return depth_path, gt_path


def process_frame(rgb_path, info):
    """Load one frame, detect candidates and save annotations.

    Args:
        rgb_path: pathlib.Path to the RGB image.
        info: dict[str, str]; parsed tree, season and time.

    Returns:
        dict or None: Frame identity, candidates, dimensions and
            read_ms/detection_ms/save_ms; None if depth or GT is missing. main adds
            total_ms.

    Raises:
        ValueError: If RGB/depth shapes differ.
        OSError: If decoding or annotation saving fails.
        cv2.error: If image processing fails.
    """
    depth_path, gt_path = get_matching_files(rgb_path, info)
    if not depth_path.exists() or not gt_path.exists():
        return None
    read_start = time.perf_counter()
    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f'RGB/depth shape mismatch: {rgb_path.name}')
    gt = load_gt(gt_path, depth.shape)
    read_ms = (time.perf_counter() - read_start) * 1000
    detect_start = time.perf_counter()
    candidates = find_apple_candidates(rgb, depth, gt)
    detection_ms = (time.perf_counter() - detect_start) * 1000
    save_start = time.perf_counter()
    annotated = draw_annotations(rgb, candidates)
    output_dir = ANNOTATED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / rgb_path.name
    Image.fromarray(annotated).save(output_path)
    save_ms = (time.perf_counter() - save_start) * 1000
    return {
        'tree': info['tree'], 'season': info['season'], 'time': info['time'],
        'apple_count': len(candidates), 'annotated_path': output_path,
        'image': rgb_path.name, 'instances': candidates,
        'image_width': rgb.shape[1], 'image_height': rgb.shape[0],
        'read_ms': read_ms, 'detection_ms': detection_ms, 'save_ms': save_ms,
    }


def find_all_frames():
    """Discover complete configured-season RGB/depth/GT triplets.

    Warns and skips unexpected names or missing companions. An absent RGB directory
    produces an empty list.

    Returns:
        list[tuple[pathlib.Path, dict]]: Frames sorted by tree and timestamp string.
    """
    frames = []
    for rgb_path in RGB_DIR.glob('*.png'):
        info = parse_rgb_filename(rgb_path)
        if info is None:
            warnings.warn(f'Unexpected RGB filename: {rgb_path.name}')
            continue
        if info['season'].lower() != SEASON.lower():
            continue
        depth_path, gt_path = get_matching_files(rgb_path, info)
        if not depth_path.exists() or not gt_path.exists():
            warnings.warn(f'Missing depth or GT for {rgb_path.name}')
            continue
        frames.append((rgb_path, info))
    return sorted(frames, key=lambda item: (item[1]['tree'], item[1]['time']))


def instance_rows(result):
    """Export components in the shared instance-CSV convention.

    confidence is intentionally blank; red_score is an uncalibrated colour statistic.

    Args:
        result: dict with image, instances, image_width and image_height.

    Yields:
        dict: One row per candidate with 1-based apple_id, inclusive xyxy box and
            mean-pixel centroid in original RGB coordinates.

    Raises:
        KeyError: If required result/candidate fields are absent.
    """
    for apple_id, apple in enumerate(result['instances'], 1):
        x, y, w, h = apple['bbox']
        cx, cy = apple['centroid']
        yield {
            'image': result['image'], 'apple_id': apple_id,
            'confidence': '', 'area_px': apple['area'],
            'branch_overlap': apple['branch_overlap'],
            'cx_px': cx, 'cy_px': cy,
            'x1': x, 'y1': y, 'x2': x + w - 1, 'y2': y + h - 1,
            'red_score': apple['red_score'],
            'image_width': result['image_width'],
            'image_height': result['image_height'],
        }


def save_csv(results):
    """Write counts, instances, timing CSVs and run metadata.

    Overwrites configured CSVs. Durations are milliseconds. Metadata records traditional
    parameters and GT full-resize alignment.

    Args:
        results: list[dict]; process_frame results with total_ms added by main.

    Returns:
        None.

    Raises:
        KeyError: If any export field, including total_ms, is missing.
        OSError: If output creation or writing fails.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with COUNTS_CSV.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tree', 'Season', 'Time', 'Apple_Count', 'Annotated_Image'])
        for result in results:
            writer.writerow([
                result['tree'], result['season'], result['time'],
                result['apple_count'], str(result['annotated_path']),
            ])

    with TIMINGS_CSV.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'apple_count', 'read_ms', 'detection_ms',
                         'save_ms', 'total_ms'])
        for result in results:
            writer.writerow([result['image'], result['apple_count'],
                             f'{result["read_ms"]:.4f}',
                             f'{result["detection_ms"]:.4f}',
                             f'{result["save_ms"]:.4f}',
                             f'{result["total_ms"]:.4f}'])

    with INSTANCES_CSV.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=INSTANCE_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerows(instance_rows(result))

    metadata = {
        'algorithm': 'apple_detector_v1', 'branch_source': 'gt',
        'annotation_color': 'orange',
        'branch_alignment': 'full_resize',
        'bbox_convention': 'inclusive xyxy',
        'centroid': 'connected component pixel mean in original RGB coordinates',
        'confidence': 'unavailable for traditional detector; blank',
        'parameters': {
            'RED_THRESHOLD': RED_THRESHOLD,
            'RED_DOMINANCE': RED_DOMINANCE,
            'MIN_APPLE_AREA': MIN_APPLE_AREA,
            'MAX_APPLE_AREA': MAX_APPLE_AREA,
            'MIN_CIRCULARITY': MIN_CIRCULARITY,
            'MAX_ASPECT_RATIO': MAX_ASPECT_RATIO,
            'BRANCH_DILATION': BRANCH_DILATION,
            'MIN_BRANCH_OVERLAP': MIN_BRANCH_OVERLAP,
            'MORPH_KERNEL_SIZE': MORPH_KERNEL_SIZE,
        },
    }
    (OUTPUT_ROOT / 'run_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )


def main():
    """Process configured frames and export traditional detector outputs.

    Configuration uses module constants, not CLI flags. Prints and returns without
    exports when no complete frames exist.

    Returns:
        None.

    Raises:
        ValueError: If RGB/depth geometry differs.
        OSError: If input reading or output writing fails.
        cv2.error: If processing fails.
    """
    frames = find_all_frames()
    if not frames:
        print(f'No matching frames found. Dataset: {DATASET}')
        return
    print(f'Processing {len(frames)} frames...')
    results = []
    for i, (rgb_path, info) in enumerate(frames, 1):
        frame_start = time.perf_counter()
        result = process_frame(rgb_path, info)
        if result is None:
            warnings.warn(f'Skipping {rgb_path.name}: missing input')
            continue
        result['total_ms'] = (time.perf_counter() - frame_start) * 1000
        results.append(result)
        print(f'[{i}/{len(frames)}] {rgb_path.name}: '
              f'{result["apple_count"]} apples | '
              f'detection={result["detection_ms"]:.2f} ms | '
              f'total={result["total_ms"]:.2f} ms')
    save_csv(results)
    print(f'Done: {len(results)} frames')
    print(f'Counts: {COUNTS_CSV}')
    print(f'Instances: {INSTANCES_CSV}')
    print(f'Timings: {TIMINGS_CSV}')
    if results:
        print(f'Mean detection: {np.mean([r["detection_ms"] for r in results]):.2f} ms/image')
        print(f'Mean total: {np.mean([r["total_ms"] for r in results]):.2f} ms/image')


if __name__ == '__main__':
    main()
