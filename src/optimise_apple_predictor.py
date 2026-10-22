"""One-factor-at-a-time traditional apple-detector sensitivity experiment.

Place alongside apple_predictor_v1.py and compare_apple_detectors.py in src/.
Run from repository root: python src/optimise_apple_predictor.py

Changes only in-memory traditional detector constants; the source, RF-DETR predictions,
branch dilation and overlap stay unchanged. No annotated PNGs are generated.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    'RED_THRESHOLD': 50,
    'RED_DOMINANCE': 1.5,
    'MIN_APPLE_AREA': 30,
    'MAX_APPLE_AREA': 5000,
    'MIN_CIRCULARITY': 0.20,
    'MAX_ASPECT_RATIO': 2.5,
    'MORPH_KERNEL_SIZE': 5,
}
# One-factor-at-a-time. No branch parameters in this list.
SWEEPS = {
    'RED_THRESHOLD': [30, 40, 50, 60],
    'RED_DOMINANCE': [1.2, 1.35, 1.5, 1.65],
    'MIN_APPLE_AREA': [10, 20, 30, 50],
    'MAX_APPLE_AREA': [2000, 5000, 8000],
    'MIN_CIRCULARITY': [0.10, 0.20, 0.30],
    'MAX_ASPECT_RATIO': [1.8, 2.5, 3.5],
    'MORPH_KERNEL_SIZE': [3, 5, 7],
}
FIELDS = ['image','apple_id','confidence','area_px','branch_overlap',
          'cx_px','cy_px','x1','y1','x2','y2','red_score','image_width','image_height']


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def tree_partition(filename, test_fraction=0.2):
    """Deterministic tree-ID grouping so observations of one tree stay together."""
    tree = filename.split('_', 1)[0]
    fraction = int(hashlib.sha256(tree.encode()).hexdigest()[:12], 16) / (16 ** 12)
    return 'test' if fraction < test_fraction else 'development'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', type=Path, default=ROOT/'src/apple_predictor_v1.py')
    parser.add_argument('--comparison', type=Path, default=ROOT/'src/compare_apple_detectors.py')
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'outputs/apple_rfdetr_instances')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/apple_optimisation')
    parser.add_argument('--max-distance', type=float, default=30)
    parser.add_argument('--split', choices=['all','by-tree'], default='by-tree',
                        help='by-tree selects on development trees, evaluates once on held-out trees')
    args = parser.parse_args()

    predictor = load_module(args.predictor.resolve(), 'apple_pred_experiment')
    comparator = load_module(args.comparison.resolve(), 'apple_compare_experiment')
    # Lock the branch configuration to the existing predictor values. Never modify them.
    branch_constants = {name:getattr(predictor,name) for name in
                        ('BRANCH_DILATION','MIN_BRANCH_OVERLAP')}
    if branch_constants != {'BRANCH_DILATION':100,'MIN_BRANCH_OVERLAP':0.05}:
        raise ValueError(f'Unexpected branch constants: {branch_constants}; adjust source or verify before testing')
    for key, value in BASELINE.items():
        if getattr(predictor,key) != value:
            raise ValueError(f'Baseline mismatch: {key}={getattr(predictor,key)} (expected {value})')

    ref_path = args.reference_dir/'apple_instances.csv'
    counts_path = args.reference_dir/'apple_counts.csv'
    reference, _ = comparator.load_instances(ref_path)
    ref_counts = comparator.load_counts(counts_path, reference)
    frames = predictor.find_all_frames()
    frames = [(path, info) for path,info in frames if path.name in ref_counts]
    if not frames:
        raise ValueError('No matching processed RGB filenames with RF-DETR reference CSVs')
    missing = sorted(set(ref_counts)-{p.name for p,_ in frames})
    if missing:
        print(f'WARNING: {len(missing)} reference frames omitted due to missing RGB/depth/GT inputs')
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    groups = {path.name: ('all' if args.split=='all' else tree_partition(path.name)) for path,_ in frames}
    if args.split=='by-tree' and not ({'development','test'} <= set(groups.values())):
        raise ValueError('Tree split has no development or no test images. Use --split all, or adjust partition logic.')

    # Load the images once to reduce repeated I/O. The baseline detector's loading and
    # exact branch-GT resize are preserved; no mask alignment is silently altered.
    loaded=[]
    for path,info in frames:
        rgb=predictor.load_rgb(path)
        dpath,gpath=predictor.get_matching_files(path,info)
        depth=predictor.load_depth(dpath)
        if rgb.shape[:2]!=depth.shape:
            raise ValueError(f'RGB/depth shape mismatch for {path.name}')
        gt=predictor.load_gt(gpath, depth.shape)
        loaded.append((path.name, info, rgb, depth, gt))
    print(f'Loaded {len(loaded)} frames. Fixed reference: {args.reference_dir}')
    print(f'Branch constants fixed: {branch_constants}; split={args.split}')

    experiments=[('baseline',None,None)]
    for parameter, values in SWEEPS.items():
        for value in values:
            if value != BASELINE[parameter]:
                experiments.append((f'{parameter}_{str(value).replace(".","p")}',parameter,value))
    summaries=[]
    cached={}
    for idx,(tag,parameter,value) in enumerate(experiments,1):
        for key,base in BASELINE.items():
            setattr(predictor,key,base)
        if parameter:
            setattr(predictor,parameter,value)
        folder=out/'runs'/tag
        folder.mkdir(parents=True,exist_ok=True)
        all_instances=[]
        all_counts=[]
        prediction_groups={}
        started=time.perf_counter()
        for filename,info,rgb,depth,gt in loaded:
            apples=predictor.find_apple_candidates(rgb,depth,gt)
            prediction_groups[filename]=[]
            result={'image':filename,'instances':apples,'image_width':rgb.shape[1], 'image_height':rgb.shape[0]}
            rows=list(predictor.instance_rows(result))
            all_instances.extend(rows)
            all_counts.append({'Tree':info['tree'],'Season':info['season'],
                               'Time':info['time'],'Apple_Count':len(apples),
                               'Annotated_Image':filename})
            for row in rows:
                prediction_groups[filename].append(row)
        save_rows(folder/'apple_instances.csv', FIELDS, all_instances)
        save_rows(folder/'apple_counts.csv',
                  ['Tree','Season','Time','Apple_Count','Annotated_Image'],all_counts)
        for segment in (['all'] if args.split=='all' else ['development','test']):
            if args.split=='by-tree' and segment=='test' and tag!='baseline':
                # Do not inspect test scores for parameter selection.
                continue
            names=[name for name in prediction_groups if groups[name]==segment]
            if not names:
                continue
            per,pairs,metrics=comparator.evaluate(names,reference,prediction_groups,args.max_distance)
            row={'run':tag,'parameter':parameter or 'baseline',
                 'value':value if parameter else '', 'split':segment, **metrics}
            summaries.append(row)
            if segment in ('development','all'):
                save_rows(folder/'per_image_results.csv',list(per[0]),per)
        print(f'[{idx}/{len(experiments)}] {tag}: '
              f'{next(r for r in reversed(summaries) if r["run"]==tag)["f1_relative_to_rfdetr"]:.4f} F1 '
              f'({time.perf_counter()-started:.1f}s)')
        cached[tag]=prediction_groups

    selection_split='development' if args.split=='by-tree' else 'all'
    choices=[r for r in summaries if r['split']==selection_split]
    # Exclude test images from selection; deterministic F1 primary, MAE secondary.
    best=sorted(choices,key=lambda r:(-r['f1_relative_to_rfdetr'],r['count_mae'],0 if r['run']=='baseline' else 1,r['run']))[0]
    if args.split=='by-tree':
        test_names=[n for n in groups if groups[n]=='test']
        for tag in ('baseline',best['run']):
            _,_,metrics=comparator.evaluate(test_names,reference,cached[tag],args.max_distance)
            summaries.append({'run':tag,'parameter':'held_out_test','value':'',
                              'split':'test',**metrics})
    save_rows(out/'all_experiments.csv',list(summaries[0]),summaries)
    # Paper-ready parameter sensitivity plots (development/all split only).
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for ax,(parameter,values) in zip(axes.flat, SWEEPS.items()):
            pts=[]
            for value in values:
                if value == BASELINE[parameter]:
                    row=next(r for r in choices if r['run']=='baseline')
                else:
                    tag=f'{parameter}_{str(value).replace(".","p")}'
                    row=next(r for r in choices if r['run']==tag)
                pts.append((float(value),row))
            pts.sort()
            for field,label in [('precision_relative_to_rfdetr','Precision'),
                                ('recall_relative_to_rfdetr','Recall'),
                                ('f1_relative_to_rfdetr','F1')]:
                ax.plot([v for v,_ in pts],[100*r[field] for _,r in pts],marker='o',label=label)
            ax.axvline(BASELINE[parameter],linestyle='--',alpha=.4)
            ax.set_title(parameter)
            ax.set_xlabel('Parameter value')
            ax.set_ylabel('RF-DETR agreement (%)')
            ax.grid(alpha=.2)
        axes.flat[-1].axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower right')
        fig.tight_layout()
        fig.savefig(out/'parameter_sensitivity.png',dpi=160)
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable: parameter_sensitivity.png not generated')
    selected={'selected_run':best['run'],'selected_parameter':best['parameter'],
              'selected_value':best['value'],'criterion':'max development F1, then min MAE',
              'branch_constants':branch_constants,'baseline_parameters':BASELINE,
              'matching_threshold_px':args.max_distance, 'split':args.split,
              'n_frames':len(loaded),'n_development':sum(x==selection_split for x in groups.values()),
              'reference_instances_sha256':hashlib.sha256(ref_path.read_bytes()).hexdigest(),
              'predictor_sha256':hashlib.sha256(args.predictor.read_bytes()).hexdigest(),
              'comparison_sha256':hashlib.sha256(args.comparison.read_bytes()).hexdigest()}
    (out/'experiment_metadata.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print('\nSelected based on development subset:',best['run'])
    print('Results:',out/'all_experiments.csv')
    print('NOTE: This is one-factor-at-a-time sensitivity, not joint/global optimisation.')
    print('NOTE: RF-DETR is pseudo-GT; existing GT mask alignment differences remain.')

if __name__=='__main__':
    main()
