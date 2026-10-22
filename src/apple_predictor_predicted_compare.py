#!/usr/bin/env python3
"""Use improved U-Net branch predictions for traditional apple detection, then compare.

Run from the repository root with the mma3001 environment:
    python src/apple_predictor_predicted_compare.py

Default workflow: load a trusted improved U-Net checkpoint, infer a branch mask for
*each summer RGB-D frame*, put the 256x256 prediction back into the central square
of the original RGB image, run the UNCHANGED traditional apple candidate detector,
export compatible prediction CSVs, and immediately compare them against the EXISTING
RF-DETR pseudo-reference CSVs. The RF-DETR model is NOT run or modified.

Optional --also-gt-control runs an equivalent traditional detection experiment with
GT branch masks using the SAME centre-square alignment. This is a more controlled
comparison of predicted vs GT branches than using historical full-resize results.

IMPORTANT: On summer training/validation images this is not an independent test of
branch-network generalisation. RF-DETR CSVs are pseudo-ground truth, not apple GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# The script must live in src/ beside these two unchanged project modules.
import apple_predictor_v1 as detector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth'
DEFAULT_REFERENCE = ROOT / 'outputs/apple_rfdetr_instances'
DEFAULT_RESULTS = ROOT / 'outputs/apple_detection_predicted'
DEFAULT_COMPARISON = ROOT / 'outputs/apple_detector_comparison_predicted'
DEFAULT_MASKS = ROOT / 'outputs/branch_predictions'


def centre_square_mask(mask_2d: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Project a square branch mask into the central square of the original image.

    Matches the RF-DETR code's ``align_branch_mask(..., 'center_crop')`` geometry:
    nearest-neighbour resize to min(height, width), then zero-pad around the square.

    Args:
        mask_2d: Binary 2D mask; 0 or positive values.
        image_shape: Original RGB (height, width) in pixels.

    Returns:
        uint8 mask of shape (height, width) with values 0 and 255.
    """
    height, width = image_shape
    if mask_2d.ndim != 2:
        raise ValueError('Branch mask must be single-channel')
    side = min(height, width)
    square = cv2.resize((mask_2d > 0).astype(np.uint8), (side, side),
                        interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = (width - side) // 2, (height - side) // 2
    canvas[y0:y0 + side, x0:x0 + side] = square * 255
    return canvas


def load_improved_model(checkpoint_path: Path, requested_device: str):
    """Load a trusted trained improved U-Net; do not re-train or download weights.

    The exact checkpoint state keys must match the ImprovedUNet implementation in
    winter_inference.py. Loading with weights_only=True avoids unrestricted pickle.

    Returns:
        (model, torch module, winter_inference module, device, input size, checkpoint info)
    """
    import torch
    import winter_inference as winter

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Improved U-Net checkpoint missing: {checkpoint_path}')
    device_str = ('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device
    if device_str == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; use --device cpu')
    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected a checkpoint dictionary with model_state_dict')
    state = checkpoint.get('model_state_dict', checkpoint)
    model = winter.ImprovedUNet()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    size = int(checkpoint.get('input_size', 256))
    return model, torch, winter, device, size, checkpoint


def make_branch_prediction(rgb_path, depth_path, model_info, threshold: float):
    """Perform improved U-Net RGB-D inference and return a 0/255 square mask."""
    model, torch, winter, device, size, _ = model_info
    input_tensor, _, _ = winter.preprocess({'rgb': rgb_path, 'depth': depth_path}, size)
    with torch.inference_mode():
        logits = model(input_tensor.unsqueeze(0).to(device))
        mask = (torch.sigmoid(logits)[0, 0] >= threshold).to(torch.uint8)
        return (mask.cpu().numpy() * 255).astype(np.uint8)


def read_predicted_mask(path: Path) -> np.ndarray:
    """Read an existing square 0/255 predicted mask; reject 0/1-only exports."""
    if not path.is_file():
        raise FileNotFoundError(f'Predicted branch mask missing: {path}')
    result = np.asarray(Image.open(path).convert('L'))
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f'Expected a square predicted mask: {path}')
    if result.max(initial=0) == 1:
        raise ValueError(f'Mask is encoded 0/1, expected 0/255: {path}')
    return (result > 127).astype(np.uint8) * 255


def detect_and_export(frames, source, args, model_info=None):
    """Run unchanged traditional filtering using predicted or aligned GT branch masks.

    Writes new CSVs and flat annotated PNGs to an experiment-specific directory.
    The original outputs/apple_detection directory is never touched.
    """
    if source not in ('predicted', 'gt'):
        raise ValueError('source must be predicted or gt')
    output_dir = args.output_dir if source == 'predicted' else args.gt_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if source == 'predicted':
        args.mask_dir.mkdir(parents=True, exist_ok=True)

    rows_count, rows_instances, rows_timings = [], [], []
    processed = 0
    for idx, (path, info, depth_path, gt_path) in enumerate(frames, start=1):
        t0 = time.perf_counter()
        rgb = detector.load_rgb(path)
        depth = detector.load_depth(depth_path)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f'Unaligned RGB/depth: {path.name}: {rgb.shape[:2]} vs {depth.shape}')
        read_ms = 1000 * (time.perf_counter() - t0)

        tb = time.perf_counter()
        if source == 'predicted':
            mask_path = args.mask_dir / f'{path.stem}_branch.png'
            if args.reuse_masks:
                mask = read_predicted_mask(mask_path)
            else:
                mask = make_branch_prediction(path, depth_path, model_info, args.threshold)
                Image.fromarray(mask, mode='L').save(mask_path)
        else:
            if not gt_path.is_file():
                raise FileNotFoundError(f'GT mask is required for the control run: {gt_path}')
            mask = (np.asarray(Image.open(gt_path).convert('L')) > 127).astype(np.uint8) * 255
        aligned = centre_square_mask(mask, rgb.shape[:2])
        branch_ms = 1000 * (time.perf_counter() - tb)

        td = time.perf_counter()
        apples = detector.find_apple_candidates(rgb, depth, aligned)
        detection_ms = 1000 * (time.perf_counter() - td)
        ts = time.perf_counter()
        annotated_path = output_dir / path.name
        Image.fromarray(detector.draw_annotations(rgb, apples)).save(annotated_path)
        save_ms = 1000 * (time.perf_counter() - ts)

        result = {'image': path.name, 'instances': apples,
                  'image_width': rgb.shape[1], 'image_height': rgb.shape[0]}
        rows_instances.extend(detector.instance_rows(result))
        rows_count.append({'Tree': info['tree'], 'Season': info['season'], 'Time': info['time'],
                           'Apple_Count': len(apples), 'Annotated_Image': str(annotated_path)})
        rows_timings.append({'image': path.name, 'apple_count': len(apples),
                             'read_ms': f'{read_ms:.4f}', 'branch_ms': f'{branch_ms:.4f}',
                             'detection_ms': f'{detection_ms:.4f}', 'save_ms': f'{save_ms:.4f}',
                             'total_ms': f'{(time.perf_counter() - t0) * 1000:.4f}'})
        processed += 1
        print(f'[{source} {idx}/{len(frames)}] {path.name}: {len(apples)} apples')

    def write_csv(filename, columns, rows):
        with (output_dir / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    write_csv('apple_counts.csv', ['Tree', 'Season', 'Time', 'Apple_Count', 'Annotated_Image'], rows_count)
    write_csv('apple_instances.csv', detector.INSTANCE_FIELDS, rows_instances)
    write_csv('image_timings.csv', ['image', 'apple_count', 'read_ms', 'branch_ms',
                                  'detection_ms', 'save_ms', 'total_ms'], rows_timings)
    meta = {
        'branch_source': 'improved_unet_predicted' if source == 'predicted' else 'ground_truth',
        'branch_alignment': 'center_crop', 'branch_mask_binary_cutoff':
        f'predicted sigmoid >= {args.threshold}' if source == 'predicted' else 'GT > 127',
        'n_images': processed, 'season': args.season,
        'reference': 'RF-DETR predictions are a fixed pseudo-reference, not apple ground truth',
        'checkpoint': str(args.checkpoint) if source == 'predicted' else None,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if source == 'predicted' and args.checkpoint.is_file() else None,
        'reused_exported_masks': bool(args.reuse_masks) if source == 'predicted' else False,
        'thresholds': {k: getattr(detector, k) for k in
                       ('RED_THRESHOLD', 'RED_DOMINANCE', 'MIN_APPLE_AREA', 'MAX_APPLE_AREA',
                        'MIN_CIRCULARITY', 'MAX_ASPECT_RATIO', 'MORPH_KERNEL_SIZE',
                        'BRANCH_DILATION', 'MIN_BRANCH_OVERLAP')},
        'notes': ['Traditional apple detector find_apple_candidates() is unchanged.',
                  'Full original RGB pixels used for apple detection and CSV coordinates.',
                  'Only the branch mask changes; both experiments use central-square placement.'],
    }
    (output_dir / 'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return processed


def run_comparison(traditional_dir: Path, output_dir: Path, args, source: str):
    """Run the existing matching/evaluation CLI, then correct its static GT-only prose.

    All metrics come from the repository's unchanged compare_apple_detectors.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'src' / 'compare_apple_detectors.py'),
               '--traditional-dir', str(traditional_dir), '--reference-dir', str(args.reference_dir),
               '--output-dir', str(output_dir), '--rgb-dir', str(args.rgb_dir),
               '--max-distance', str(args.max_distance)]
    print('\nComparing against fixed RF-DETR reference:\n', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    # The existing comparison generator has static claims about the ORIGINAL GT-only
    # detector. Replace those paragraphs in this run's generated report only.
    report_path = output_dir / 'evaluation_report.md'
    if report_path.is_file():
        paragraphs = report_path.read_text(encoding='utf-8').split('\n\n')
        updated = []
        for p in paragraphs:
            if p.startswith('Both configured branch sources are GT, not predicted'):
                if source == 'predicted':
                    p = ('RF-DETR reference detections were generated previously using its configured GT branch masks. '
                         'In THIS experiment, the traditional detector uses branches predicted by the improved U-Net. '
                         'These masks are placed in the central square using nearest-neighbour scaling, matching RF-DETR '
                         'geometry (although branch-mask contents differ). Dilation=100 pixels and overlap >=0.05 remain fixed.')
                else:
                    p = ('The traditional control run and RF-DETR reference use GT branch masks with central-square '
                         'alignment. Traditional GT is binarised >127, matching RF-DETR. Branch dilation=100 pixels '
                         'and overlap >=0.05 remain fixed.')
            elif p.startswith('Traditional configuration is recorded in outputs/apple_detection/run_metadata.json.'):
                p = (f'This traditional experiment records its parameters, branch source and alignment in '
                     f'{traditional_dir}/run_metadata.json. The apple filter itself is unchanged; confidence is blank '
                     'and red_score is not a model probability.')
            elif p.startswith('Implementations: src/apple_predictor_v1.py'):
                p = ('Traditional candidate filtering uses the existing src/apple_predictor_v1.py, driven by '
                     'src/apple_predictor_predicted_compare.py; RF-DETR reference data are loaded from saved CSVs. '
                     'No RF-DETR inference is run during this comparison.')
            elif p.startswith('Differences combine detector behaviour, branch alignment, mask binarization'):
                p = ('Differences reflect traditional-vs-RF-DETR detector behaviour, branch source, validity filtering '
                     'and missed/merged/split detections. RF-DETR pseudo-reference instances are not verified apple '
                     'ground truth. The summer branch model was trained/selected using summer images, so predicted '
                     'mask evaluation on those images is NOT independent generalisation testing. No detector parameter '
                     'tuning is performed by this script.')
            elif p.startswith('From repository root:\n```powershell'):
                p = (f'From repository root:\n```powershell\nconda activate mma3001\n'
                     f'python src/apple_predictor_predicted_compare.py\n```')
            updated.append(p)
        report_path.write_text('\n\n'.join(updated), encoding='utf-8')

    with (output_dir / 'summary_metrics.csv').open(newline='', encoding='utf-8-sig') as f:
        row = next(csv.DictReader(f))
    keys = ('precision_relative_to_rfdetr', 'recall_relative_to_rfdetr', 'f1_relative_to_rfdetr',
            'count_mae', 'count_rmse', 'mean_centroid_error_px', 'mean_matched_bbox_iou')
    print('\n===== ' + source.upper() + ' branch comparison =====')
    for key in keys:
        print(f'{key}: {row.get(key, "(not provided)")}')
    print('Results:', output_dir / 'summary_metrics.csv')
    return row


def build_frames(args):
    """Discover RGB-D pairs; GT is optional unless the GT control is requested."""
    frames = []
    for path in sorted(args.rgb_dir.glob('*.png')):
        info = detector.parse_rgb_filename(path)
        if not info or info['season'].lower() != args.season.lower():
            continue
        depth_path = args.depth_dir / f'{info["tree"]}_{info["season"]}_D-{info["time"]}.png'
        gt_path = args.gt_dir / f'{info["tree"]}_{info["season"]}_GT-{info["time"]}.png'
        if not depth_path.is_file():
            warnings.warn(f'Missing depth, skipping {path.name}')
            continue
        if args.branch_source == 'gt' or args.also_gt_control:
            if not gt_path.is_file():
                warnings.warn(f'Missing GT, skipping {path.name} to retain paired comparison')
                continue
        frames.append((path, info, depth_path, gt_path))
    if args.max_images is not None:
        frames = frames[:args.max_images]
    if not frames:
        raise RuntimeError(f'No complete RGB-D frames found in {args.rgb_dir}')
    return frames


def main(argv=None):
    """Infer branches, detect apples, write CSVs and run the existing comparator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', default='Summer')
    parser.add_argument('--rgb-dir', type=Path, default=ROOT / 'data/Organised/Summer/RGB')
    parser.add_argument('--depth-dir', type=Path, default=ROOT / 'data/Organised/Summer/D')
    parser.add_argument('--gt-dir', type=Path, default=ROOT / 'data/Organised/Summer/GT')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--reference-dir', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--comparison-dir', type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument('--gt-output-dir', type=Path, default=ROOT / 'outputs/apple_detection_gt_aligned')
    parser.add_argument('--gt-comparison-dir', type=Path, default=ROOT / 'outputs/apple_detector_comparison_gt_aligned')
    parser.add_argument('--mask-dir', type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--branch-source', choices=('predicted', 'gt'), default='predicted')
    parser.add_argument('--reuse-masks', action='store_true', help='Use existing 0/255 square masks instead of running U-Net')
    parser.add_argument('--also-gt-control', action='store_true', help='Also run centre-aligned GT branch control')
    parser.add_argument('--threshold', type=float, default=0.5, help='Improved U-Net sigmoid threshold')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--max-distance', type=float, default=30, help='Centroid matching distance, pixels')
    parser.add_argument('--max-images', type=int, default=None, help='Optional small smoke test, not a full evaluation')
    parser.add_argument('--skip-comparison', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    if args.branch_source == 'gt' and args.also_gt_control:
        parser.error('--also-gt-control is redundant with --branch-source gt')
    if not args.skip_comparison:
        for name in ('apple_counts.csv', 'apple_instances.csv'):
            if not (args.reference_dir / name).is_file():
                parser.error(f'Missing saved RF-DETR reference CSV: {args.reference_dir / name}')

    frames = build_frames(args)
    model_info = None
    if args.branch_source == 'predicted' and not args.reuse_masks:
        model_info = load_improved_model(args.checkpoint, args.device)
        print('Loaded improved U-Net checkpoint:', args.checkpoint)
        print('Model input:', model_info[4], 'x', model_info[4], '| device:', model_info[3])

    print('RGB-D frames:', len(frames), '| branch source:', args.branch_source)
    detect_and_export(frames, args.branch_source, args, model_info)
    if not args.skip_comparison:
        output_dir = args.output_dir if args.branch_source == 'predicted' else args.gt_output_dir
        comparison_dir = args.comparison_dir if args.branch_source == 'predicted' else args.gt_comparison_dir
        run_comparison(output_dir, comparison_dir, args, args.branch_source)

    if args.also_gt_control:
        print('\nRunning same-settings GT-aligned control...')
        detect_and_export(frames, 'gt', args)
        if not args.skip_comparison:
            run_comparison(args.gt_output_dir, args.gt_comparison_dir, args, 'gt')
            print('\nCompare summary_metrics.csv in both directories to isolate branch-source effects.')
    print('\nCompleted. Original outputs/apple_detection and RF-DETR outputs remain unchanged.')


if __name__ == '__main__':
    main()
