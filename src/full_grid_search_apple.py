"""Full Cartesian grid search for traditional apple-detector parameters.

Place in src/ beside apple_predictor_v1.py and compare_apple_detectors.py.
Run from the repository root: python src/full_grid_search_apple.py

All 5,184 combinations are evaluated against the *existing* RF-DETR instance CSV.
Only traditional colour/component/morphology constants vary. Branch-region
constants, GT alignment, matching threshold and RF-DETR files are unchanged.
A resume-safe all_experiments.csv is appended after each successful configuration.
By default tune on development trees, then compare the chosen model with the
baseline on separate held-out trees. No annotated PNGs or per-run CSVs are made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    'RED_THRESHOLD': [30, 40, 50, 60],
    'RED_DOMINANCE': [1.2, 1.35, 1.5, 1.65],
    'MIN_APPLE_AREA': [10, 20, 30, 50],
    'MAX_APPLE_AREA': [2000, 5000, 8000],
    'MIN_CIRCULARITY': [0.10, 0.20, 0.30],
    'MAX_ASPECT_RATIO': [1.8, 2.5, 3.5],
    'MORPH_KERNEL_SIZE': [3, 5, 7],
}
BASELINE = {
    'RED_THRESHOLD': 50, 'RED_DOMINANCE': 1.5,
    'MIN_APPLE_AREA': 30, 'MAX_APPLE_AREA': 5000,
    'MIN_CIRCULARITY': 0.20, 'MAX_ASPECT_RATIO': 2.5,
    'MORPH_KERNEL_SIZE': 5,
}
SUMMARY_FIELDS = [
    'run_id', 'split', *GRID,
    'matched_images', 'threshold_px', 'reference_total', 'traditional_total',
    'reference_mean_count', 'traditional_mean_count',
    'count_mae', 'count_rmse', 'mean_signed_count_error',
    'count_exact_match_rate', 'count_within_one_rate', 'total_count_relative_error',
    'tp_relative_to_rfdetr', 'fp_relative_to_rfdetr', 'fn_relative_to_rfdetr',
    'precision_relative_to_rfdetr', 'recall_relative_to_rfdetr',
    'f1_relative_to_rfdetr', 'image_macro_f1',
    'mean_centroid_error_px', 'median_centroid_error_px',
    'p90_centroid_error_px', 'mean_matched_bbox_iou',
    'median_matched_bbox_iou', 'detection_seconds',
]


def load_module(path, alias):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_partition(filename):
    tree = filename.split('_', 1)[0]
    fraction = int(hashlib.sha256(tree.encode()).hexdigest()[:12], 16) / 16**12
    return 'test' if fraction < 0.2 else 'development'


def combinations():
    keys = list(GRID)
    for index, values in enumerate(itertools.product(*(GRID[k] for k in keys))):
        yield f'grid_{index:04d}', dict(zip(keys, values))


def append_row(path, row):
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_rows(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError('Existing result CSV has incompatible columns; choose a new --output-dir')
        return list(reader)


def numeric(row, key, maximize=True):
    try:
        value = float(row[key])
        if not math.isfinite(value):
            return float('-inf')
    except (ValueError, TypeError, KeyError):
        return float('-inf')
    return value if maximize else -value


def score(row, metric):
    if metric == 'count_bias':
        return -abs(float(row['mean_signed_count_error']))
    mapping = {
        'f1': ('f1_relative_to_rfdetr', True),
        'precision': ('precision_relative_to_rfdetr', True),
        'recall': ('recall_relative_to_rfdetr', True),
        'mae': ('count_mae', False),
        'rmse': ('count_rmse', False),
        'centroid': ('mean_centroid_error_px', False),
        'bbox_iou': ('mean_matched_bbox_iou', True),
    }
    key, maximize = mapping[metric]
    return numeric(row, key, maximize)


def select_best(rows, criterion):
    # Prefer the exact baseline on tied scores.
    return sorted(rows, key=lambda row: (-score(row, criterion),
                  0 if all(float(row[k]) == v for k, v in BASELINE.items()) else 1,
                  row['run_id']))[0]


def evaluate_configuration(params, frames, names, predictor, comparator, reference, threshold):
    for name, value in params.items():
        setattr(predictor, name, value)  # no source-file writes
    predictions = {}
    start = time.perf_counter()
    for filename, info, rgb, depth, branch in frames:
        apples = predictor.find_apple_candidates(rgb, depth, branch)
        record = {'image': filename, 'instances': apples,
                  'image_width': rgb.shape[1], 'image_height': rgb.shape[0]}
        predictions[filename] = list(predictor.instance_rows(record))
    duration = time.perf_counter() - start
    _, _, metrics = comparator.evaluate(names, reference, predictions, threshold)
    return metrics, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', type=Path, default=ROOT/'src/apple_predictor_v1.py')
    parser.add_argument('--comparison', type=Path, default=ROOT/'src/compare_apple_detectors.py')
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'outputs/apple_rfdetr_instances')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/apple_grid_search')
    parser.add_argument('--max-distance', type=float, default=30.0)
    parser.add_argument('--select-by', choices=['f1','precision','recall','mae','rmse',
                                                'centroid','bbox_iou','count_bias'], default='f1')
    parser.add_argument('--split', choices=['by-tree','all'], default='by-tree')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='For testing only: stop after this many new combinations')
    parser.add_argument('--no-cache', action='store_true',
                        help='Reload images per configuration to reduce RAM requirements')
    args = parser.parse_args()
    predictor_path = args.predictor.resolve()
    comparison_path = args.comparison.resolve()
    ref_path = (args.reference_dir/'apple_instances.csv').resolve()
    count_path = (args.reference_dir/'apple_counts.csv').resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_module(predictor_path, '_apple_grid_predictor')
    comparator = load_module(comparison_path, '_apple_grid_comparator')

    locked = {name: getattr(predictor, name) for name in ('BRANCH_DILATION','MIN_BRANCH_OVERLAP')}
    if locked != {'BRANCH_DILATION': 100, 'MIN_BRANCH_OVERLAP': 0.05}:
        raise ValueError(f'Branch settings are not the expected fixed baseline: {locked}')
    for name, value in BASELINE.items():
        if getattr(predictor, name) != value:
            raise ValueError(f'Unexpected original {name}={getattr(predictor,name)}; expected {value}')

    reference, _ = comparator.load_instances(ref_path)
    reference_counts = comparator.load_counts(count_path, reference)
    raw_frames = [(path, info) for path,info in predictor.find_all_frames()
                  if path.name in reference_counts]
    if not raw_frames:
        raise ValueError('No frames common to traditional inputs and RF-DETR reference CSVs')
    groups = {path.name: ('all' if args.split == 'all' else tree_partition(path.name))
              for path, info in raw_frames}
    split = 'development' if args.split == 'by-tree' else 'all'
    names = sorted(name for name, part in groups.items() if part == split)
    test_names = sorted(name for name, part in groups.items() if part == 'test')
    if not names or (args.split == 'by-tree' and not test_names):
        raise ValueError('Development/test partition empty; use --split all or adjust split')

    def load_one(path, info):
        rgb = predictor.load_rgb(path)
        depth_path, gt_path = predictor.get_matching_files(path, info)
        depth = predictor.load_depth(depth_path)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f'RGB/depth mismatch: {path.name}')
        branch = predictor.load_gt(gt_path, depth.shape)
        return path.name, info, rgb, depth, branch

    def get_frames(target_split):
        eligible = [(path, info) for path,info in raw_frames if groups[path.name] == target_split]
        return [load_one(path, info) for path,info in eligible]

    frames = None if args.no_cache else get_frames(split)
    metadata = {
        'grid': GRID, 'baseline': BASELINE, 'locked_branch_parameters': locked,
        'matching_distance_px': args.max_distance, 'split': args.split,
        'reference_instances_sha256': file_hash(ref_path),
        'reference_counts_sha256': file_hash(count_path),
        'predictor_sha256': file_hash(predictor_path),
        'comparison_sha256': file_hash(comparison_path),
        'development_images_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        'n_development_images': len(names), 'n_test_images': len(test_names),
    }
    meta_path = out/'grid_metadata.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
        if previous != metadata:
            raise ValueError('Grid settings or inputs changed. Use a new --output-dir to avoid mixing results.')
    else:
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    result_path = out/'all_experiments.csv'
    completed = {r['run_id'] for r in load_rows(result_path)}
    total = math.prod(len(values) for values in GRID.values())
    print(f'Full grid: {total} combinations; already completed: {len(completed)}')
    print(f'Development images: {len(names)}; held-out images: {len(test_names)}')
    print('Fixed branch settings:', locked, '| fixed RF-DETR reference:', ref_path)
    done_now = 0
    try:
        for index, (run_id, params) in enumerate(combinations(), start=1):
            if run_id in completed:
                continue
            current_frames = get_frames(split) if args.no_cache else frames
            metrics, seconds = evaluate_configuration(params, current_frames, names,
                                predictor, comparator, reference, args.max_distance)
            append_row(result_path, {'run_id':run_id, 'split':split, **params, **metrics,
                                     'detection_seconds': round(seconds, 4)})
            done_now += 1
            if done_now == 1 or done_now % 10 == 0:
                print(f'[{index}/{total}] {run_id}: F1={metrics["f1_relative_to_rfdetr"]:.4f}, '
                      f'MAE={metrics["count_mae"]:.3f}, precision={metrics["precision_relative_to_rfdetr"]:.4f}, '
                      f'recall={metrics["recall_relative_to_rfdetr"]:.4f}', flush=True)
            if args.max_runs is not None and done_now >= args.max_runs:
                break
    finally:
        for name, value in BASELINE.items():
            setattr(predictor, name, value)

    rows = load_rows(result_path)
    if not rows:
        print('No completed experiments yet')
        return
    objectives = ['f1','precision','recall','mae','rmse','centroid','bbox_iou','count_bias']
    rankings = []
    for objective in objectives:
        ordered = sorted(rows, key=lambda r: (-score(r,objective),r['run_id']))
        for rank, row in enumerate(ordered, 1):
            rankings.append({'metric':objective,'rank':rank,**row})
    rank_path = out/'metric_rankings.csv'
    with rank_path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric','rank',*SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerows(rankings)
    best = select_best(rows, args.select_by)
    with (out/'best_by_metric.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric',*SUMMARY_FIELDS])
        writer.writeheader()
        for metric in objectives:
            writer.writerow({'metric':metric,**select_best(rows,metric)})

    # Only after an exhaustive search, evaluate the selected model on held-out trees.
    if len(rows) == total and args.split == 'by-tree':
        heldout_path = out/'heldout_comparison.csv'
        if not heldout_path.exists():
            heldout_frames = get_frames('test')
            with heldout_path.open('w',newline='',encoding='utf-8') as f:
                writer = csv.DictWriter(f,fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                for label,params in [('baseline',BASELINE),('selected', {k: type(BASELINE[k])(float(best[k])) for k in GRID})]:
                    metrics,secs = evaluate_configuration(params,heldout_frames,test_names,
                                           predictor,comparator,reference,args.max_distance)
                    writer.writerow({'run_id':label,'split':'test',**params,**metrics,
                                     'detection_seconds':round(secs,4)})
            for name, value in BASELINE.items():
                setattr(predictor,name,value)

    print(f'Completed {len(rows)}/{total}. Selection objective: {args.select_by}.')
    print(f'Best so far: {best["run_id"]}; F1={float(best["f1_relative_to_rfdetr"]):.4f}; '
          f'MAE={float(best["count_mae"]):.3f}')
    print('All metrics CSV:', result_path)
    print('Best of each metric:', out/'best_by_metric.csv')
    print('Rankings CSV:', rank_path)
    if len(rows) < total:
        print('Partial results only; re-run identical command to resume.')
    print('NOTE: RF-DETR is pseudo-GT, not verified apple annotations.')
    print('NOTE: Existing branch-mask alignment differences have NOT been corrected.')


if __name__ == '__main__':
    main()
