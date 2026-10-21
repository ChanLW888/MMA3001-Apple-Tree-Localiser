"""Traditional branch-guided red-apple detector with RF-DETR-compatible instance CSV.

Place in the repository's src/ directory and run from repository root:
    python src/apple_detector_v1_with_instances.py
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
ANNOTATED_DIR = OUTPUT_ROOT / 'annotated'
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
    match = re.fullmatch(
        r'(?P<tree>R\d+N\d+)_(?P<season>Summer|Winter)_RGB-(?P<time>.+)\.png',
        path.name,
        re.IGNORECASE,
    )
    return match.groupdict() if match else None


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def load_depth(path):
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def load_gt(path, target_shape):
    gt = np.asarray(Image.open(path).convert('L'))
    if gt.shape != target_shape:
        gt = cv2.resize(
            gt, (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return gt


def detect_red_mask(rgb, depth):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return ((r > RED_THRESHOLD) & (r > g * RED_DOMINANCE)
            & (r > b * RED_DOMINANCE) & (depth > 0)).astype(np.uint8)


def clean_red_mask(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def create_branch_zone(gt):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BRANCH_DILATION, BRANCH_DILATION)
    )
    return cv2.dilate((gt > 0).astype(np.uint8), kernel)


def find_apple_candidates(rgb, depth, gt):
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
        if perimeter == 0:
            continue
        # Preserve the original filtering rule for a comparable baseline.
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        local_branch = branch_zone[y:y + h, x:x + w]
        overlap_pixels = np.count_nonzero((component > 0) & (local_branch > 0))
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


def draw_annotations(rgb, candidates):
    """Preserve original output style; RGB array passed to PIL on save."""
    annotated = rgb.copy()
    for i, apple in enumerate(candidates, 1):
        x, y, w, h = apple['bbox']
        cx, cy = apple['centroid']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(annotated, str(i), (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f'Apples: {len(candidates)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated


def get_matching_files(rgb_path, info):
    depth_path = DEPTH_DIR / f'{info["tree"]}_{info["season"]}_D-{info["time"]}.png'
    gt_path = GT_DIR / f'{info["tree"]}_{info["season"]}_GT-{info["time"]}.png'
    return depth_path, gt_path


def process_frame(rgb_path, info):
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
    output_dir = ANNOTATED_DIR / info['tree']
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
    """RF-DETR-compatible first 11 columns; blank confidence is intentional."""
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
