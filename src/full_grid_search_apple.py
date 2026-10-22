"""Full Cartesian grid search for traditional apple-detector parameters.

Place in src/ beside apple_predictor_v1.py and compare_apple_detectors.py.
Run from the repository root: python src/full_grid_search_apple.py

All 5,184 combinations are evaluated against the *existing* RF-DETR instance CSV.
Only traditional colour/component/morphology constants vary. Branch-region
constants, GT alignment, matching threshold and RF-DETR files are unchanged.
A resumable all_experiments.csv is appended after each successful configuration.
See main for fingerprint coverage and the held-out objective-change limitation.
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
    """Load a Python file under a supplied import alias.

    Executes module-level statements and registers the module in sys.modules;
    callers must use trusted, import-safe files.

    Args:
        path: pathlib.Path to a Python source file.
        alias: str module name used in sys.modules.

    Returns:
        ModuleType: Executed module.

    Raises:
        FileNotFoundError: If path is not a file.
        Exception: Exceptions raised by the imported module propagate.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    """Fingerprint a complete file for run provenance.

    Args:
        path: pathlib.Path to a readable file; bytes are read into memory.

    Returns:
        str: Hexadecimal SHA-256 digest.

    Raises:
        OSError: If reading fails.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_partition(filename):
    """Assign every frame of one tree to the same deterministic split.

    Args:
        filename: str basename whose prefix before the first underscore is a
            tree ID. No filename validation is performed.

    Returns:
        str: ``test`` when the SHA-256-derived fraction is below 0.2,
        otherwise ``development``. The observed split need not be exactly 20%.
    """
    tree = filename.split('_', 1)[0]
    fraction = int(hashlib.sha256(tree.encode()).hexdigest()[:12], 16) / 16**12
    return 'test' if fraction < 0.2 else 'development'


def combinations():
    """Enumerate the Cartesian product of the current module GRID.

    Grid insertion order determines run IDs. The current seven parameter lists
    produce 4 * 4 * 4 * 3 * 3 * 3 * 3 = 5,184 configurations. Branch dilation
    and overlap are absent from the grid and remain fixed.

    Yields:
        tuple[str, dict]: Zero-based grid_NNNN ID and detector constants.
        Each call produces new parameter dictionaries.
    """
    keys = list(GRID)
    for index, values in enumerate(itertools.product(*(GRID[k] for k in keys))):
        yield f'grid_{index:04d}', dict(zip(keys, values))


def append_row(path, row):
    """Append one configuration and its metrics to the result CSV.

    Args:
        path: pathlib.Path; parent directory must exist.
        row: dict with SUMMARY_FIELDS-compatible keys. Missing values are blank.
            A header is written only when the file is empty.

    Returns:
        None.

    Raises:
        ValueError: If row contains unexpected columns.
        OSError: If opening or writing fails.
    """
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_rows(path):
    """Read resumable results and require the exact current column order.

    Args:
        path: pathlib.Path to all_experiments.csv.

    Returns:
        list[dict[str, str]]: Raw CSV rows, or an empty list for an absent file.
        Run IDs and row values are not otherwise validated here.

    Raises:
        ValueError: If the header differs from SUMMARY_FIELDS.
        OSError: If an existing file cannot be read.
    """
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError('Existing result CSV has incompatible columns; choose a new --output-dir')
        return list(reader)


def numeric(row, key, maximize=True):
    """Convert a metric into a score where larger is preferred.

    Args:
        row: Mapping of metric names to numeric values or numeric strings.
        key: str metric column.
        maximize: bool; negate finite values when False.

    Returns:
        float: Oriented score; negative infinity for absent, invalid or
        nonfinite values, independent of optimisation direction.
    """
    try:
        value = float(row[key])
        if not math.isfinite(value):
            return float('-inf')
    except (ValueError, TypeError, KeyError):
        return float('-inf')
    return value if maximize else -value


def score(row, metric):
    """Map a supported optimisation objective to a larger-is-better score.

    Args:
        row: Mapping containing evaluation metric columns.
        metric: One of f1, precision, recall, mae, rmse, centroid, bbox_iou,
            count_bias. Errors/distances are minimised, agreement maximised,
            and signed bias minimised in absolute magnitude.

    Returns:
        float: Oriented score; most invalid metrics become negative infinity.

    Raises:
        KeyError: If metric is unknown or count_bias lacks its input column.
        ValueError: If count_bias cannot be converted to float.
        TypeError: If count_bias has an incompatible value type.

    Note:
        Unlike other objectives, count_bias does not reject nonfinite values.
    """
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
    """Select the best row with deterministic baseline tie preference.

    Args:
        rows: Nonempty sequence of result dictionaries with run_id, BASELINE
            parameter columns and the requested metric.
        criterion: str objective accepted by score.

    Returns:
        dict: Original winning row. Equal scores prefer exact baseline values,
        then lexicographically smallest run_id.

    Raises:
        IndexError: If rows is empty.
        KeyError: If an objective or required column is missing or unknown.
        ValueError: If a parameter cannot be converted to float.
        TypeError: If a numeric value has an incompatible type.
    """
    return sorted(rows, key=lambda row: (-score(row, criterion),
                  0 if all(float(row[k]) == v for k, v in BASELINE.items()) else 1,
                  row['run_id']))[0]


def evaluate_configuration(params, frames, names, predictor, comparator, reference, threshold):
    """Evaluate cached synthetic or real frames without exporting images.

    Mutates predictor constants in memory; this helper does not restore them.
    The main search restores baseline constants in a finally block.

    Args:
        params: Mapping of detector constant names to parameter values.
        frames: Iterable of (filename, info, rgb, depth, branch) tuples. RGB
            has shape [H, W, 3]; aligned depth and branch masks have shape [H, W].
            The info element is unused here.
        names: Nonempty reusable collection of exact basenames to evaluate.
        predictor: Module-like object exposing detector and instance_rows APIs.
        comparator: Module-like object exposing evaluate.
        reference: Mapping of basenames to saved RF-DETR instance records.
        threshold: float inclusive centroid tolerance in original RGB pixels.

    Returns:
        tuple[dict, float]: Aggregate comparison metrics and elapsed seconds
        for detection/instance conversion, excluding comparison evaluation.

    Raises:
        ValueError: If spatial shapes or comparison tolerance are invalid.
        ZeroDivisionError: If names is empty.
        KeyError: If candidate/export fields are missing.
    """
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
    """Run or resume traditional parameter tuning against saved RF-DETR results.

    Reads command-line options through argparse; use --help for paths,
    selection objectives, by-tree/all splits, caching and partial-run options.

    Returns:
        None. Appends all_experiments.csv and rewrites best_by_metric.csv and
        metric_rankings.csv. Writes grid_metadata.json and, only after a complete
        by-tree search, heldout_comparison.csv if that file does not already exist.

    Raises:
        ValueError: If baseline/fixed branch settings differ, common inputs or
            partitions are empty, or resume metadata/CSV columns are incompatible.
        FileNotFoundError: If configured source or reference files are missing.
        OSError: If image/CSV reading or result writing fails.
        SystemExit: For argparse help or invalid arguments.

    Note:
        Branch dilation is fixed at 100 pixels and minimum overlap at 0.05.
        Complete traditional RGB/depth/GT triplets are intersected with reference
        count basenames, preserving zero-count images. GT is stretched to RGB
        geometry. Images are cached in RAM unless --no-cache reloads each run.
        RF-DETR CSV predictions are pseudo-reference, never new inference.

        Metrics include micro precision/recall/F1, image macro F1, signed and
        absolute counting errors, RMSE, count agreement rates, conditional centroid
        distances in pixels and inclusive-box IoU. By-tree mode tunes development
        trees and evaluates only the selected configuration and baseline on test
        trees; --split all provides no held-out estimate.

        Resume checks grid, source/reference hashes and development-name hash,
        then skips recorded run IDs. Image bytes, environment versions and test
        names are not fully fingerprinted. Timing depends on machine/load. Source
        docstring edits also change hashes and require a fresh output directory.

        Known limitation: --select-by is not included in metadata, and an existing
        heldout_comparison.csv is never refreshed. A changed objective can reuse
        stale held-out results. Use a separate output directory per objective;
        retain original results. Partial rankings are not exhaustive optima.
    """
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
        """Load one aligned RGB/depth/branch frame for evaluation.

            Args:
                path: pathlib.Path to an RGB image.
                info: dict with tree, season and time used to locate companions.

            Returns:
                tuple: Basename, info, RGB [H, W, 3], depth [H, W] and resized
                branch GT [H, W]; depth retains stored units.

            Raises:
                ValueError: If RGB and depth dimensions differ.
                OSError: If image decoding fails.
                KeyError: If companion metadata is missing.
        """
        rgb = predictor.load_rgb(path)
        depth_path, gt_path = predictor.get_matching_files(path, info)
        depth = predictor.load_depth(depth_path)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f'RGB/depth mismatch: {path.name}')
        branch = predictor.load_gt(gt_path, depth.shape)
        return path.name, info, rgb, depth, branch

    def get_frames(target_split):
        """Load all discovered frames belonging to a split.

            Args:
                target_split: str; development, test or all, matching groups.

            Returns:
                list[tuple]: Loaded frame tuples in discovery order; empty if none.

            Raises:
                ValueError: If a frame has inconsistent RGB/depth shapes.
                OSError: If image reading fails.
        """
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
