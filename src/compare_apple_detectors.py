"""Compare traditional predictions with RF-DETR pseudo-ground truth.

Matches exact basenames and original-image centroids from saved CSVs. One-to-one
matching maximises valid pairs before minimising distance. Agreement is not
independently verified accuracy. Runs no detector inference. Generated report prose
includes fixed historical statements that this program does not verify.

Example:
    python src/compare_apple_detectors.py --max-distance 30 --thresholds 10 20 30 40 50
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'image', 'apple_id', 'cx_px', 'cy_px', 'x1', 'y1', 'x2', 'y2', 'area_px', 'branch_overlap', 'confidence'}


def read_csv(path):
    """Read headers and rows from a possibly BOM-prefixed CSV.

    Args:
        path: str or pathlib.Path to UTF-8 CSV.

    Returns:
        tuple[list[str], list[dict[str, str]]]: Header names and unconverted rows.

    Raises:
        ValueError: If no header exists.
        OSError: If input reading fails.
    """
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'{path}: missing header')
        return reader.fieldnames, list(reader)


def load_instances(path):
    """Validate and group inclusive-coordinate instance records.

    IDs are positive and unique per image. Finite nonnegative coordinates must contain
    the centroid; image bounds are checked later by main. Confidence may be blank or
    NaN; finite scores must be in [0, 1]. Duplicate geometry is retained.

    Args:
        path: str or pathlib.Path; CSV containing REQUIRED columns.

    Returns:
        tuple[dict[str, list[dict]], list[dict]]: Grouped records and retained
            duplicate-geometry warnings.

    Raises:
        ValueError: If headers, IDs, geometry, area, overlap or confidence are
            invalid.
        OSError: If CSV reading fails.
    """
    fields, rows = read_csv(path)
    if REQUIRED - set(fields):
        raise ValueError(f'{path}: missing columns {sorted(REQUIRED-set(fields))}')
    grouped = defaultdict(list)
    ids, geometries = set(), set()
    duplicates = []
    for line, row in enumerate(rows, 2):
        try:
            if not row['image'] or Path(row['image']).name != row['image']:
                raise ValueError('image must be an exact basename')
            key = (row['image'], int(row['apple_id']))
            if key in ids or key[1] < 1:
                raise ValueError('duplicate or invalid apple_id')
            ids.add(key)
            for column in REQUIRED - {'image', 'apple_id', 'confidence'}:
                row[column] = float(row[column])
                if not np.isfinite(row[column]):
                    raise ValueError(f'non-finite {column}')
            if row['confidence']:
                score = float(row['confidence'])
                if not np.isnan(score) and (not np.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('invalid confidence')
            if not (0 <= row['x1'] <= row['cx_px'] <= row['x2'] and
                    0 <= row['y1'] <= row['cy_px'] <= row['y2'] and
                    row['area_px'] > 0 and 0 <= row['branch_overlap'] <= 1):
                raise ValueError('invalid geometry/area/overlap')
            geometry = (row['image'], *(row[k] for k in ('x1','y1','x2','y2','cx_px','cy_px')))
            if geometry in geometries:
                duplicates.append(dict(image=row['image'], apple_id=row['apple_id'], issue='duplicate geometry; retained'))
            geometries.add(geometry)
            grouped[row['image']].append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}:{line}: {exc}') from exc
    return grouped, duplicates


def load_counts(path, instances):
    """Validate processed-image counts against instance rows.

    Traditional identity comes from Annotated_Image; RF-DETR uses image. Counts
    distinguish zero detections from unprocessed images.

    Args:
        path: str or pathlib.Path; traditional or RF-DETR counts CSV.
        instances: Mapping[str, list[dict]] grouped by exact image basename.

    Returns:
        dict[str, int]: Nonnegative counts, including processed zero-detection
            frames.

    Raises:
        ValueError: If headers, integer counts, unique names or instance/count
            agreement fail.
        OSError: If CSV reading fails.
    """
    fields, rows = read_csv(path)
    traditional = 'Apple_Count' in fields
    required = {'Annotated_Image','Apple_Count'} if traditional else {'image','kept_apple_count'}
    if required - set(fields):
        raise ValueError(f'{path}: missing count columns {required-set(fields)}')
    counts = {}
    for line, row in enumerate(rows, 2):
        image = row['Annotated_Image'].replace('\\','/').split('/')[-1] if traditional else row['image']
        count = int(row['Apple_Count'] if traditional else row['kept_apple_count'])
        if not image or image in counts or count < 0:
            raise ValueError(f'{path}:{line}: duplicate image or invalid count')
        counts[image] = count
        if count != len(instances.get(image, [])):
            raise ValueError(f'{path}:{line}: instance/count discrepancy for {image}')
    if set(instances)-set(counts):
        raise ValueError(f'{path}: instances without processed-image count records')
    return counts


def match_centroids(reference, traditional, threshold):
    """Maximise valid one-to-one pairs, then minimise summed distance.

    Hungarian assignment adds unmatched dummy columns. A penalty larger than any valid
    total prioritises cardinality over distance. RF-DETR is pseudo-ground truth, not
    independently verified apple annotation.

    Args:
        reference: Sequence[dict] with RF-DETR cx_px and cy_px.
        traditional: Sequence[dict] with traditional cx_px and cy_px.
        threshold: float; finite nonnegative inclusive Euclidean tolerance in
            original-image pixels.

    Returns:
        list[tuple[int, int, float]]: Reference index, traditional index and
            distance; empty if either sequence is empty.

    Raises:
        ValueError: If threshold is negative/nonfinite or assignment costs are
            invalid.
    """
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError('threshold must be finite and nonnegative')
    if not reference or not traditional:
        return []
    r = np.array([[a['cx_px'], a['cy_px']] for a in reference])
    p = np.array([[a['cx_px'], a['cy_px']] for a in traditional])
    distances = np.linalg.norm(r[:,None,:]-p[None,:,:], axis=2)
    # One unmatched penalty exceeds any sum of valid distances, preserving cardinality.
    penalty = (min(len(r),len(p))+1)*(threshold+1)
    costs = np.full((len(r), len(p)+len(r)), penalty)
    costs[:,:len(p)] = np.where(distances <= threshold, distances, penalty*2)
    rr, pp = linear_sum_assignment(costs)
    return [(int(i),int(j),float(distances[i,j])) for i,j in zip(rr,pp)
            if j < len(p) and distances[i,j] <= threshold]


def bbox_iou(a,b):
    """Measure overlap of two inclusive xyxy pixel boxes.

    The +1 terms count both boundary pixels. IoU is descriptive, not a centroid matching
    criterion.

    Args:
        a: dict with numeric x1, y1, x2, y2; valid nonempty box.
        b: dict with the same inclusive-coordinate fields.

    Returns:
        float: Intersection/union in [0, 1] for valid boxes.

    Raises:
        KeyError: If a coordinate is missing.
    """
    intersection = max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1'])+1)*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1'])+1)
    area = lambda r: (r['x2']-r['x1']+1)*(r['y2']-r['y1']+1)
    return intersection/(area(a)+area(b)-intersection)


def agreement(tp, fp, fn):
    """Compute micro agreement relative to RF-DETR.

    Error labels express disagreement with model predictions, not verified real-world
    errors.

    Args:
        tp: int; nonnegative accepted match count.
        fp: int; nonnegative unmatched traditional detections.
        fn: int; nonnegative unmatched reference detections.

    Returns:
        dict[str, int | float]: Reference-relative counts, precision, recall and F1;
            undefined ratios return zero.
    """
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(tp_relative_to_rfdetr=tp, fp_relative_to_rfdetr=fp,
                fn_relative_to_rfdetr=fn, precision_relative_to_rfdetr=precision,
                recall_relative_to_rfdetr=recall,
                f1_relative_to_rfdetr=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def evaluate(images, ref, pred, threshold):
    """Summarise fixed detections over common processed images.

    Missing instance keys imply zero detections. Count error is traditional minus
    reference. Distance/IoU statistics condition on accepted matches; absent samples
    yield None. Global agreement is micro; image_macro_f1 averages per-image F1.

    Args:
        images: Nonempty reusable collection[str] of common exact basenames,
            including zero-count frames.
        ref: Mapping[str, list[dict]] of RF-DETR pseudo-reference records.
        pred: Mapping[str, list[dict]] of traditional records.
        threshold: float; finite nonnegative inclusive tolerance in pixels.

    Returns:
        tuple[list[dict], list[dict], dict]: Per-image metrics, matched/unmatched
            pairs and aggregate summary.

    Raises:
        ValueError: If matching tolerance is invalid.
        ZeroDivisionError: If images is empty.
    """
    per_image, pairs, distances = [], [], []
    for image in sorted(images):
        r,p = ref.get(image,[]), pred.get(image,[])
        matches = match_centroids(r,p,threshold)
        ri,pj = {i for i,j,d in matches},{j for i,j,d in matches}
        ds = [d for i,j,d in matches]
        error = len(p)-len(r)
        per_image.append(dict(image=image, rfdetr_reference_count=len(r), traditional_count=len(p),
            signed_count_error=error, absolute_count_error=abs(error),
            **agreement(len(matches),len(p)-len(matches),len(r)-len(matches)),
            mean_centroid_error_px=float(np.mean(ds)) if ds else None,
            median_centroid_error_px=float(np.median(ds)) if ds else None))
        def pair(i,j,d,status):
            """Build a matched or unmatched record for the current image.

            Args:
                i: int or None; reference index, absent for unmatched prediction.
                j: int or None; prediction index, absent for unmatched reference.
                d: float or str; accepted distance in pixels, or empty string.
                status: str; matched, unmatched_reference or unmatched_traditional.

            Returns:
                dict: Image, IDs, distance, conditional inclusive-box IoU and
                    status.

            Raises:
                IndexError: If a supplied index is outside the current records.
            """
            return dict(image=image,reference_apple_id=r[i]['apple_id'] if i is not None else '',
                traditional_apple_id=p[j]['apple_id'] if j is not None else '',
                centroid_distance_px=d, bbox_iou=bbox_iou(r[i],p[j]) if i is not None and j is not None else '',status=status)
        pairs.extend(pair(i,j,d,'matched') for i,j,d in matches)
        pairs.extend(pair(i,None,'','unmatched_reference') for i in range(len(r)) if i not in ri)
        pairs.extend(pair(None,j,'','unmatched_traditional') for j in range(len(p)) if j not in pj)
        distances.extend(ds)
    errors = np.array([r['signed_count_error'] for r in per_image])
    nr=sum(len(ref.get(i,[])) for i in images)
    npred=sum(len(pred.get(i,[])) for i in images)
    matched_ious=[row['bbox_iou'] for row in pairs if row['status']=='matched']
    summary=dict(
        matched_images=len(images), threshold_px=threshold,
        reference_total=nr, traditional_total=npred,
        reference_mean_count=nr/len(images), traditional_mean_count=npred/len(images),
        count_mae=float(np.mean(np.abs(errors))),
        count_rmse=float(np.sqrt(np.mean(errors**2))),
        mean_signed_count_error=float(np.mean(errors)),
        count_exact_match_rate=float(np.mean(errors == 0)),
        count_within_one_rate=float(np.mean(np.abs(errors) <= 1)),
        total_count_relative_error=(npred-nr)/nr if nr else None,
        **agreement(len(distances),npred-len(distances),nr-len(distances)),
        image_macro_f1=float(np.mean([r['f1_relative_to_rfdetr'] for r in per_image])),
        mean_centroid_error_px=float(np.mean(distances)) if distances else None,
        median_centroid_error_px=float(np.median(distances)) if distances else None,
        p90_centroid_error_px=float(np.percentile(distances,90)) if distances else None,
        mean_matched_bbox_iou=float(np.mean(matched_ious)) if matched_ious else None,
        median_matched_bbox_iou=float(np.median(matched_ious)) if matched_ious else None,
    )
    return per_image,pairs,summary


def write_csv(path,rows,fields=None):
    """Write dictionaries using explicit or inferred columns.

    Args:
        path: str or pathlib.Path; parent must exist.
        rows: list[dict]; records compatible with fields.
        fields: Sequence[str] or None; explicit order, required for empty rows.

    Returns:
        None.

    Raises:
        IndexError: If rows is empty and fields is absent/empty.
        ValueError: If rows have unexpected columns.
        OSError: If writing fails.
    """
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Compare saved CSVs without rerunning either detector.

    Writes metrics, sensitivity, coverage, examples, SHA-256 manifest and annotation
    inventory. Defaults to apple_rfdetr_instances and apple_detection under outputs.
    Current source is not proof of historical CSV provenance. Embedded report prose is
    fixed program data, not verified experiment history.

    Returns:
        None.

    Raises:
        SystemExit: If CLI arguments are invalid or --help is requested.
        ValueError: If CSVs, counts, geometry, coverage or tolerance are invalid.
        OSError: If inputs or report/plot outputs cannot be accessed.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=ROOT/'outputs/apple_rfdetr_instances')
    parser.add_argument('--traditional-dir',type=Path,default=ROOT/'outputs/apple_detection')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/apple_detector_comparison')
    parser.add_argument('--rgb-dir',type=Path,default=ROOT/'data/Organised/Summer/RGB')
    parser.add_argument('--max-distance',type=float,default=30)
    parser.add_argument('--thresholds',type=float,nargs='+',default=[10,20,30,40,50])
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    ref,rd=load_instances(args.reference_dir/'apple_instances.csv')
    pred,pd=load_instances(args.traditional_dir/'apple_instances.csv')
    rc=load_counts(args.reference_dir/'apple_counts.csv',ref)
    pc=load_counts(args.traditional_dir/'apple_counts.csv',pred)
    # Counts distinguish processed zero-detection frames from absent instance rows.
    images=set(rc)&set(pc)
    if not images:
        raise ValueError('No exact matching processed image filenames')
    images = sorted(images)
    coverage=[dict(image=i,status='matched' if i in images else 'reference_only' if i in rc else 'traditional_only') for i in sorted(set(rc)|set(pc))]
    write_csv(out/'image_coverage.csv',coverage)
    write_csv(out/'duplicate_detections.csv',[dict(method=m,**d) for m,ds in [('RF-DETR',rd),('traditional',pd)] for d in ds],['method','image','apple_id','issue'])
    dimensions=Counter()
    for image in set(rc)|set(pc):
        path=args.rgb_dir/image
        if not path.is_file():
            raise ValueError(f'Missing source RGB: {path}')
        with Image.open(path) as im:
            width,height=im.size
        dimensions[f'{width}x{height}']+=1
        for row in ref.get(image,[])+pred.get(image,[]):
            if row['x2']>=width or row['y2']>=height:
                raise ValueError(f'Out-of-image bbox: {image}, apple {row["apple_id"]}')
    per,pairs,summary=evaluate(images,ref,pred,args.max_distance)
    # Include unmatched images explicitly, with undefined cross-method metrics.
    for c in coverage:
        if c['status']!='matched':
            per.append(dict(image=c['image'],rfdetr_reference_count=rc.get(c['image']),traditional_count=pc.get(c['image'])))
    write_csv(out/'per_image_results.csv',per,list(per[0]))
    write_csv(out/'summary_metrics.csv',[summary])
    write_csv(out/'matched_instances.csv',pairs,['image','reference_apple_id','traditional_apple_id','centroid_distance_px','bbox_iou','status'])
    sensitivity=[]
    for t in args.thresholds:
        _,_,s=evaluate(images,ref,pred,t)
        sensitivity.append(s)
    write_csv(out/'threshold_sensitivity.csv',sensitivity)
    matched_per=[r for r in per if r['image'] in images]
    fig,ax=plt.subplots()
    x=[r['rfdetr_reference_count'] for r in matched_per]
    y=[r['traditional_count'] for r in matched_per]
    ax.scatter(x,y,color='darkorange',alpha=.6,label='apple_detector_v1 vs RF-DETR')
    lim=max(x+y+[1]); ax.plot([0,lim],[0,lim],color='teal',label='Equal counts')
    ax.set(xlabel='RF-DETR reference count (apples)',ylabel='apple_detector_v1 count (apples)',title='Count agreement with RF-DETR pseudo-ground truth')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'count_comparison.png'); plt.close(fig)
    for values,name,xlabel in [([r['signed_count_error'] for r in matched_per],'count_error_distribution','Traditional minus RF-DETR count (apples)'),([r['centroid_distance_px'] for r in pairs if r['status']=='matched'],'centroid_error_distribution','Accepted centroid distance (pixels)')]:
        fig,ax=plt.subplots(); ax.hist(values,bins=25,color='darkorange')
        ax.set(xlabel=xlabel,ylabel='Frequency',title='Agreement with RF-DETR pseudo-ground truth')
        fig.tight_layout(); fig.savefig(out/(name+'.png')); plt.close(fig)
    fig,ax=plt.subplots()
    for key,label in [('precision_relative_to_rfdetr','Precision'),('recall_relative_to_rfdetr','Recall'),('f1_relative_to_rfdetr','F1')]:
        ax.plot(args.thresholds,[s[key] for s in sensitivity],marker='o',label=label)
    ax.set(xlabel='Maximum centroid distance (pixels)',ylabel='Agreement with RF-DETR (0–1)',title='Matching sensitivity: fixed detector predictions'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'threshold_sensitivity.png'); plt.close(fig)
    example_dir=out/'comparison_examples'; example_dir.mkdir(exist_ok=True)
    selections={'strong_agreement':max(matched_per,key=lambda r:r['f1_relative_to_rfdetr']),
                'overcounting':max(matched_per,key=lambda r:r['signed_count_error']),
                'undercounting':min(matched_per,key=lambda r:r['signed_count_error']),
                'large_localisation_error':max(matched_per,key=lambda r:r['mean_centroid_error_px'] or -1)}
    for label,row in selections.items():
        image=row['image']; im=Image.open(args.rgb_dir/image).convert('RGB'); draw=ImageDraw.Draw(im)
        for rows,color in [(ref.get(image,[]),'cyan'),(pred.get(image,[]),'orange')]:
            for a in rows:
                draw.rectangle(tuple(a[k] for k in ('x1','y1','x2','y2')),outline=color,width=1)
                cx,cy=a['cx_px'],a['cy_px']; draw.ellipse((cx-2,cy-2,cx+2,cy+2),fill=color)
        for i,j,d in match_centroids(ref.get(image,[]),pred.get(image,[]),args.max_distance):
            a,b=ref[image][i],pred[image][j]; draw.line((a['cx_px'],a['cy_px'],b['cx_px'],b['cy_px']),fill='white')
        draw.rectangle((0,0,im.width,32),fill='black')
        draw.text((4,2),'RF-DETR reference: cyan | apple_detector_v1: orange',fill='white')
        draw.text((4,16),label+' | matched centroids: white lines',fill='white')
        im.save(example_dir/(label+'.png'))
    sources=[args.reference_dir/'apple_instances.csv',args.reference_dir/'apple_counts.csv',args.traditional_dir/'apple_instances.csv',args.traditional_dir/'apple_counts.csv',ROOT/'src/apple_predictor_v1.py',ROOT/'src/apple_segmenter_v1.py']
    manifest={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    (out/'input_manifest.json').write_text(json.dumps(dict(sha256=manifest,arguments={k:str(v) for k,v in vars(args).items()},image_dimensions=dimensions),indent=2))
    annotations=[]
    for path in (ROOT/'data').rglob('*.json') if (ROOT/'data').is_dir() else []:
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data,dict) and 'annotations' in data:
                annotations.append(dict(path=str(path.relative_to(ROOT)),images=len(data.get('images',[])),annotations=len(data['annotations']),categories=data.get('categories',[])))
        except (ValueError,UnicodeError):
            pass
    (out/'annotation_inventory.json').write_text(json.dumps(annotations,indent=2))
    reference_widths=[r['x2']-r['x1']+1 for image in images for r in ref.get(image,[])]
    median_width=float(np.median(reference_widths)) if reference_widths else None
    report=['# Apple detector comparison',
        'RF-DETR predictions are pseudo-ground-truth reference annotations, not verified ground truth. Agreement does not establish real-world apple detection or localisation accuracy.',
        f'Exact filename intersection: {len(images)} images. Reference processed: {len(rc)}; traditional processed: {len(pc)}. Unmatched images: {len(coverage)-len(images)}; see image_coverage.csv. Dimensions: {dict(dimensions)}.',
        'Implementations: src/apple_predictor_v1.py (apple_detector_v1) and src/apple_segmenter_v1.py (RFDETRSegMedium). Both read data/Organised/Summer/RGB. Traditional requires valid depth; RF-DETR does not.',
        'Both configured branch sources are GT, not predicted U-Net masks. Traditional uses grayscale >0 and full-image nearest-neighbour resizing; RF-DETR uses >127, central square placement, then dilation. Both use ellipse kernel diameter 100 pixels and overlap >=0.05. These are different regions despite shared mask files; results are an as-configured comparison, not a controlled detector-only comparison.',
        'Traditional configuration is recorded in outputs/apple_detection/run_metadata.json. Red threshold 50; dominance 1.5; morphology ellipse 5; area 30–5000 pixels; circularity >=0.20; aspect ratio <=2.5. MIN_RED_AREA=15 is unused in existing code. Export adds no filtering and leaves confidence blank; red_score is separate.',
        'RF-DETR source configuration: checkpoint outputs/rfdetr_apple_tree/checkpoint_best_total.pth; confidence >=0.20; instance mask threshold >0.50; minimum area 1; largest-branch-component filtering disabled. Historical run provenance cannot be proven from CSV alone; these settings describe the current source. Existing predictions were not changed or rerun.',
        f'Coordinates: original RGB pixel coordinates, origin top left; inclusive xyxy boxes; mask/component mean centroids. Traditional conversion x2=x+w-1, y2=y+h-1. Principal matching maximizes the number of pairs within inclusive distance <= {args.max_distance:g} px, then minimizes total Euclidean distance using Hungarian assignment and dummy unmatched columns. Box IoU uses inclusive pixel areas and is descriptive only.',
        'Metrics are micro-aggregated over matched processed images, including zeros established by counts CSVs. Undefined precision/recall/F1 denominators return zero; no accepted-distance samples yield blank mean/median. Count error is traditional minus RF-DETR.',
        '\n| Metric | Value |\n|---|---|']
    report.extend(f'| {k} | {v} |' for k,v in summary.items())
    report.append('\n**Interpretation:** count MAE and RMSE measure count agreement, not apple matching; centroid error and matched-box IoU are conditional on accepted matches. F1 is reference-relative rather than independently verified accuracy. Exact-match rate is the fraction of images with identical counts, even when their locations differ.')
    report.extend(['\n## Matching threshold sensitivity',f'Fixed predictions; median reference box width {median_width if median_width is not None else "unavailable"} px. The default 30 px is a preselected image-space tolerance, not an independently calibrated physical distance. 10–50 px spans stricter to more permissive tolerances. No threshold is selected by maximum F1.',
        '| Threshold px | Precision | Recall | F1 | Matches | Mean distance px |\n|---|---|---|---|---|---|'])
    report.extend(f"| {s['threshold_px']} | {s['precision_relative_to_rfdetr']:.4f} | {s['recall_relative_to_rfdetr']:.4f} | {s['f1_relative_to_rfdetr']:.4f} | {s['tp_relative_to_rfdetr']} | {s['mean_centroid_error_px']} |" for s in sensitivity)
    report.append('\n## Representative cases')
    report.extend(f"- {label}: {r['image']}; reference={r['rfdetr_reference_count']}, traditional={r['traditional_count']}, signed error={r['signed_count_error']}, F1 agreement={r['f1_relative_to_rfdetr']:.3f}, mean accepted distance={r['mean_centroid_error_px']}." for label,r in selections.items())
    report.extend([f'\nDuplicate geometry rows retained: RF-DETR {len(rd)}, traditional {len(pd)}. Duplicate IDs are rejected. Distinct but overlapping boxes are retained, since overlap alone does not establish a duplicate apple. See duplicate_detections.csv.',
        '## Limitations and conclusions',
        'Differences combine detector behaviour, branch alignment, mask binarization, depth validity and possible merged/split apples. Count agreement can hide localisation errors. RF-DETR can miss, merge, split or hallucinate apples, so FP/FN labels only express disagreement with that reference. Accepted-distance errors are conditional on the matching threshold and cannot measure true accuracy. Images may be correlated orchard frames; no independence or statistical generalisation is claimed. No detector parameters were tuned on these evaluation results. Branch dilation/overlap sensitivity was not run; it would change predictions and must be a separately labelled experiment, ideally using common aligned branch masks and held-out human annotations.',
        f'Human-annotation candidates: {annotations}. COCO annotation authorship and correspondence to orchard evaluation images need verification. Use annotation_inventory.json and evaluate separately on human-labelled held-out images; avoid training/evaluation leakage. Existing branch GT masks label branches, not apple instances.',
        '## Reproduction',
        'From repository root:\n```powershell\nconda run -n treeid python -m unittest discover -s test -p "test_apple*.py" -v\nconda run -n treeid python src/apple_predictor_v1.py\nconda run -n treeid python src/compare_apple_detectors.py --max-distance 30 --thresholds 10 20 30 40 50\n```',
        'The RF-DETR CSVs are fixed inputs; input_manifest.json records SHA-256 fingerprints. Install/use existing numpy, scipy, matplotlib, Pillow and OpenCV dependencies. Generated outputs and raw datasets are ignored by Git.'])
    report.insert(2, 'No runtime or timing comparison is included. All metrics describe detection counts, localisation agreement, or input coverage.')
    (out/'evaluation_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Unmatched processed images: {len(coverage)-len(images)}; duplicate geometry: {len(rd)+len(pd)}')

if __name__=='__main__':
    main()
