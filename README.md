# Apple localisation for MMA3001

This Numerical Methods and Machine Learning project investigates RGB-D tree-branch segmentation and 2D apple localisation in orchard images. It compares a pixel MLP, a basic ResNet-34 U-Net and an extended U-Net for branch segmentation, alongside a traditional red-component detector and RF-DETR apple instance segmentation.

For an orchard robot, image-space fruit candidates and structural branch masks
could inform later sensing, obstacle reasoning and harvesting-system development.
This repository studies that perception stage. It does not implement a picking
controller, collision-safe robot motion, calibrated 3D apple locations or the
three-dimensional skeleton output described in the brief's suggested reconstruction
topic. Confirm that this narrower project scope meets the teaching team's agreement.

The engineering objective is reproducible image-space apple candidates and branch predictions that could support later 3D reconstruction. **The standalone traditional and RF-DETR detectors default to branch GT.** The new `apple_predictor_predicted_compare.py` driver connects improved U-Net predictions to traditional detection, with an optional aligned-GT control. Depth is a validity filter in traditional detection, not a calibrated source of 3D apple coordinates. RF-DETR detection does not read depth.

## Repository map

For a short assessment route, use the [reviewer starting point](docs/source/reviewer_start.md).
It separates checks requiring no orchard files from full experiment reproduction.

| Location | Maintained content |
|---|---|
| [environment.yml](environment.yml) | Exported `mma3001` Conda dependencies |
| [rfdetr_environment.yml](rfdetr_environment.yml) | Exported `rfdetr` Conda dependencies |
| [src/README.md](src/README.md) | Preparation, branch training, apple detection and comparison scripts |
| [test/README.md](test/README.md) | Independent synthetic unit tests and actual execution report |
| [src/winter_inference.py](src/winter_inference.py) | Qualitative branch predictions and forward-pass timing with three checkpoints |
| [data/README.md](data/README.md) | Orchard PNG naming, organised layout and COCO split requirements |
| [outputs/README.md](outputs/README.md) | Output directories, CSV conventions and interpretation |
| [docs/README.md](docs/README.md) | Sphinx project guide, static source API and HTML build instructions |

Datasets, checkpoints, generated results, caches and docs/_build/ are excluded from source control. The submission HTML in docs/html/ is not ignored. The folder READMEs are maintained documentation, not generated outputs.

## Environment and hardware

Two exported Conda environments are used. The environment specifications are included in the repository root. **Run commands from the repository root**, and use the appropriate environment for each stage.

| Workflow | Environment | Specification |
|---|---|---|
| Branch segmentation (MLP and U-Nets), traditional apple detection, evaluation, grid search, and tests | `mma3001` | [`environment.yml`](environment.yml) |
| RF-DETR training and instance-segmentation inference | `rfdetr` | [`rfdetr_environment.yml`](rfdetr_environment.yml) |

### Create the MMA3001 environment

```powershell
conda env create -f environment.yml
conda activate mma3001
```

Use this environment for the branch model training, traditional detection, comparison, and other non-RF-DETR experiments documented below. For example:

```powershell
python src/apple_predictor_v1.py
python src/compare_apple_detectors.py --max-distance 30 --thresholds 10 20 30 40 50
```

### Create the RF-DETR environment

```powershell
conda env create --name rfdetr -f rfdetr_environment.yml
conda activate rfdetr
```

Use this environment for RF-DETR inference and, when needed, training:

```powershell
python src/apple_segmenter_v1.py
python src/apple_segmenter_train.py
```

The detectors run **separately** and write their own result files. To compare them, first generate or obtain their saved prediction CSVs, then switch to `mma3001` and run the comparison script. You do **not** need both model stacks installed in the same environment.

### Environment verification and hardware

After creating each environment, you can confirm the interpreter and key imports:

```powershell
conda activate mma3001
python --version
python -c "import numpy, cv2, PIL, torch, scipy; print('MMA3001 imports OK')"
conda activate rfdetr
python --version
python -c "import torch, rfdetr; print('RF-DETR imports OK')"
```

These are checks to run, **not results already verified on a clean machine**. Conda YAML exports document the packages in the author's environments but may require compatible operating systems, GPU drivers, CUDA runtimes, or package sources on another machine. No minimum RAM, VRAM, or GPU model has been established. Branch training scripts select CUDA when available or fall back to CPU; RF-DETR device selection is managed by its library. Dataset images and trained checkpoints are supplied separately, not by the environment specifications. Pretrained weights may be downloaded on first use if not already cached.

## Dataset inputs

Run commands from the repository root. Input data must be obtained separately; this repository does not establish its redistribution rights or calibrated depth units.

```text
data/
  Dataset (With Summer GT)/       # flat organiser input
  Organised/
    Summer/
      RGB/<tree>_Summer_RGB-<time>.png
      D/<tree>_Summer_D-<time>.png
      GT/<tree>_Summer_GT-<time>.png
    Winter/
      RGB/<tree>_Winter_RGB-<time>.png
      D/<tree>_Winter_D-<time>.png
      GT/                         # may be empty
  apple tree.v1i.coco-segmentation/
    train/_annotations.coco.json  # images alongside JSON
    export/_annotations.coco.json
  apple_tree_clean/
    train/_annotations.coco.json
    valid/_annotations.coco.json
```

For example, `R10N12_Summer_RGB-13-36-51.png`, `R10N12_Summer_D-13-36-51.png` and `R10N12_Summer_GT-13-36-51.png` identify one frame. Apple scripts require the `R<digits>N<digits>` tree convention; the organiser accepts a broader tree token. Matching filenames do not prove spatial alignment.

Branch training centre-crops RGB/depth to a square and resizes to 256 pixels by default. RGB uses ImageNet mean/std after division by 255; depth is converted to 8-bit grayscale and divided by 255. Inputs are float32 `[B, 4, H, W]`, targets `[B, 1, H, W]` with branch=1 for GT >127. GT is not resized and must already match the input grid and centred field of view. The depth channel is not measured in established physical units. See [data and preprocessing](docs/source/data_and_preprocessing.md).

To organise flat orchard files (copies files and may overwrite destinations):

```powershell
python src/dataset_organiser.py
```

The organiser uses Windows-style relative paths and runs filesystem operations even if imported. RF-DETR COCO preparation is a separate workflow:

```powershell
python src/preprocess_roboflow_dataset.py
```

It merges category names beginning with `apple-` into class ID 1, retains negative images, maps `export` to `valid`, and preserves separate apple instances. Missing image copies leave COCO records behind, so inspect warnings before training. The presence of COCO files does not establish independently verified orchard evaluation annotations.

## Branch segmentation

The following commands train models; they are reproduction instructions, not actions executed for this documentation task:

```powershell
python src/train_baseline.py --data-root data/Organised
python src/train_unet.py --data-root data/Organised
python src/train_improved_unet.py --data-root data/Organised
```

Each accepts `--help`, `--output-dir`, `--input-size`, `--epochs`, `--batch-size`, `--seed`, `--val-fraction`, `--threshold` and other script-specific options. Use `--no-pretrained` for U-Nets when ImageNet initialisation is not desired. The MLP uses weighted BCE (positive weight 12) and Adam. Basic U-Net uses plain BCE and Adam. The extended U-Net adds ASPP, attention gates, residual decoding, paired flips and normalised-RGB jitter; it uses plain BCE, AdamW and a validation-Dice plateau scheduler.

All select the best checkpoint by validation Dice on a seeded shuffled summer sample split (20% validation by default). This is model-selection validation, not an independent test or a guaranteed held-out-tree split. Seeded shuffling does not guarantee bitwise deterministic CUDA execution.

For winter inference with checkpoints produced by the commands above:

```powershell
python src/winter_inference.py --data-root data/Organised --mlp-checkpoint outputs/pixel_mlp_crop_trainval/pixel_mlp_best.pth --unet-checkpoint outputs/resnet34_unet_crop_trainval_output/resnet34_unet_best.pth --improved-checkpoint outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth
```

These are expected output paths, not bundled checkpoint files. Supply the actual checkpoint locations if earlier runs used other directories. Use only trusted checkpoints. Keep `--input-size` and `--threshold` consistent with training; input size is not automatically inferred from checkpoints. Winter inference needs RGB-D pairs, not GT, and produces qualitative masks and model-only timings rather than winter accuracy metrics.

## Traditional apple detection

```powershell
python src/apple_predictor_v1.py
```

Configuration is through module constants, with summer defaults. Processing selects red-dominant pixels with depth >0, applies elliptical opening/closing, then filters connected components by area, aspect ratio, circularity and branch-zone overlap. Defaults include red >50, red dominance 1.5, area 30–5000 pixels, circularity >=0.20, aspect ratio <=2.5, morphology diameter 5, branch dilation diameter 100 and overlap >=0.05. The mean red score `R/(G+B+1)` is descriptive, not confidence and not an additional rejection threshold.

Branch GT is converted to grayscale, stretched over the full RGB grid using nearest-neighbour interpolation, and interpreted as foreground >0. RGB and depth must have matching spatial dimensions. Unexpected names or missing companions are skipped with warnings. An empty/missing RGB folder returns without producing new results.

Depth validity applies to the initial red-pixel mask. Subsequent closing can fill
non-red or invalid-depth pixels; final component pixels are not individually
guaranteed to satisfy the initial colour/depth tests.

## Predicted versus GT branch experiments

The [experiment guide](docs/source/predicted_branches.md) covers every CLI option,
alignment, timings and cache limitations. In `mma3001`, with an existing improved
U-Net checkpoint and saved RF-DETR counts/instances CSVs:

```powershell
python src/apple_predictor_predicted_compare.py --help
python src/apple_predictor_predicted_compare.py --checkpoint outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth --also-gt-control
```

Supply the actual checkpoint path if a historical run used another directory.
Fresh predictions use checkpoint `input_size` (fallback 256), training-compatible
RGB-D preprocessing and sigmoid >=0.5. Predicted masks are **not GT**. They are
expanded into the original frame's central square; apple detection uses full-frame
RGB/depth. The control uses GT >127 with that same projection, differing from the
standalone traditional detector's GT >0/full-frame resize.

Default outputs are `outputs/apple_detection_predicted`,
`outputs/apple_detector_comparison_predicted` and square masks in
`outputs/branch_predictions`. The control writes `outputs/apple_detection_gt_aligned`
and `outputs/apple_detector_comparison_gt_aligned`. RF-DETR remains a fixed
pseudo-reference and is not run. Use distinct directories for new experiments;
CSVs and matching PNGs are overwritten. `--reuse-masks` skips model loading but
does not verify cached masks against the requested checkpoint/threshold.
Summer training/validation overlap prevents an independent generalisation claim.
Source defaults do not prove historical reference settings.

## RF-DETR apple detection

With a previously trained checkpoint at `outputs/rfdetr_apple_tree/checkpoint_best_total.pth`:

```powershell
python src/apple_segmenter_v1.py
```

The script loads `RFDETRSegMedium`, applies confidence >=0.20, instance mask >0.50, minimum area 1 pixel and branch overlap >=0.05. It assumes an apple-only checkpoint; there is no class-ID filter. Branch masks are thresholded >127 and, by default, expanded into the central square of the original image before dilation (diameter 100). This geometry differs from traditional detection. Largest-branch-component filtering is disabled by default.

`BRANCH_SOURCE = "gt"` is the default. Optional `"predicted"` mode reads externally supplied `outputs/branch_predictions/<RGB-stem>_branch.png` masks encoded 0/255; 0/1 PNG values fail the >127 cutoff. Winter benchmark mask filenames/layout do not directly match this convention. This RF-DETR script makes no branch-network call. The predicted comparison driver exports compatible square masks; winter exports require adaptation. Confirm field-of-view geometry before interpreting detections.

RF-DETR training, when intentionally required, is separate:

```powershell
python src/apple_segmenter_train.py
```

It requires cleaned COCO train/valid data and requests 50 epochs at batch size 1. Importing this training module checks dataset paths and prints configuration. Avoid importing preparation/training scripts as passive libraries.

## Compare saved detections

```powershell
python src/compare_apple_detectors.py --max-distance 30 --thresholds 10 20 30 40 50
```

Defaults read `apple_counts.csv` and `apple_instances.csv` from `outputs/apple_detection` and `outputs/apple_rfdetr_instances`, plus source summer RGB images. Options include `--traditional-dir`, `--reference-dir`, `--rgb-dir` and `--output-dir`. This command reads existing predictions; it does not run detectors.

The comparison uses exact basename intersection, retaining zero-detection frames via counts CSVs. Inclusive pixel boxes use `(x1, y1, x2, y2)`; centroids are mean component/mask pixel coordinates with top-left origin, x right and y down. Hungarian assignment first maximises pairs within inclusive Euclidean tolerance, then minimises distance. IoU is descriptive and does not select matches. Signed count error is traditional minus RF-DETR.

****RF-DETR predictions are pseudo-ground truth, not independently verified apple annotations.**** Precision/recall/F1 describe reference agreement. Accepted centroid distances and box IoUs condition on matching. Shared GT dependence, different mask cutoffs/alignment, traditional depth filtering and correlated orchard frames prevent a controlled detector-only or generalisation claim. Matching sensitivity changes pairing tolerance on fixed predictions, not detector parameters.

## Expected outputs

| Producer | Default location and files |
|---|---|
| MLP training | `outputs/pixel_mlp_crop_trainval/`: `pixel_mlp_best.pth`, `history.csv`, `val_metrics.csv`, `visualisations/` |
| Basic U-Net training | `outputs/resnet34_unet_crop_trainval_output/`: `resnet34_unet_best.pth` and corresponding CSVs/visualisations |
| Extended U-Net training | `outputs/improved_resnet34_unet_output/`: `improved_resnet34_unet_best.pth` and corresponding CSVs/visualisations |
| Winter inference | `outputs/winter_inference_benchmark/`: per-model masks, overlays, comparisons; `per_image_timings.csv`, `summary.csv`, `benchmark_metadata.json` |
| Traditional detector | `outputs/apple_detection/`: flat annotation PNGs, `apple_counts.csv`, `apple_instances.csv`, `image_timings.csv`, `run_metadata.json` |
| RF-DETR inference | `outputs/apple_rfdetr_instances/`: `annotated/`, `masks/`, `branch_zones/`, counts/instances/timings CSVs |
| Comparison | `outputs/apple_detector_comparison/`: metrics, matches, coverage, duplicates, sensitivity, plots/examples, `input_manifest.json`, `annotation_inventory.json`, `evaluation_report.md` |

See [output schemas](outputs/README.md) for CSV meanings and timing boundaries. These paths describe current source defaults, which can differ from existing historical output folders. Historical experiments were not rerun. Saved files were inspected and preservation checks performed; the final audit records their provenance and validation limits. Reusing output directories overwrites named files; partial RF-DETR runs recreate CSVs rather than append to them and may leave older PNGs.

## Reproduction and verification

1. Record your source revision, environment package versions, device, data provenance and checkpoint fingerprints.

2. Check image triplets, GT field of view and COCO split independence. Keep evaluation human annotations separate from training if measuring real accuracy.

3. Run preparation only when needed, then branch training and optional winter inference separately from apple detection.

4. Use the configured local RF-DETR checkpoint and run each detector intentionally. Keep its input masks and source settings with the outputs.

5. Compare saved CSVs and retain the input manifest. Current constants alone cannot prove how historical CSVs were produced.

Run the independent synthetic software tests in `mma3001`:

```powershell
conda activate mma3001
python -m unittest discover -s test -p "test_*.py" -v
```

The suite requires no datasets, checkpoints, network, GPU or RF-DETR inference.
It tests detection, CSV export/validation, comparison metrics and grid-search helpers.
See [test instructions](test/README.md) and the [actual report](test/test_report.txt).
These checks establish software behaviour on synthetic inputs, not model accuracy.

## Full grid-search optimisation

`src/full_grid_search_apple.py` enumerates 5,184 traditional configurations.
Branch dilation (100 pixels), minimum branch overlap (0.05) and saved RF-DETR
pseudo-reference predictions remain fixed. By default, SHA-256 tree grouping
separates development and held-out frames; `--split all` removes held-out evaluation.

```powershell
conda run -n mma3001 python src/full_grid_search_apple.py --help
```

An intentional optimisation run uses the same script without `--help`; it requires
orchard inputs and saved reference CSVs and is not part of automated testing.
Selection supports F1, precision, recall, MAE, RMSE, centroid error, box IoU and
absolute count bias. Results append per configuration and compatible runs resume.
See [options](src/README.md) and [output files](outputs/README.md).

Known limitation: changing `--select-by` in an existing output directory can reuse
stale `heldout_comparison.csv`. Use a separate output directory for each objective.
Source docstring changes alter resume fingerprints; retain historical outputs and
use a new directory after this documentation update. Image bytes and environment
versions are not fully fingerprinted, so metadata alone cannot ensure reproducibility.

## HTML documentation

The Sphinx site includes all 12 maintained `src/` modules, using static AST extraction of signatures and existing docstrings. No project modules are imported. Open [submitted HTML](docs/html/index.html) directly in a browser; keep the entire `docs/html/` directory together for submission. It is outside the ignored build directory.

Rebuild from the repository root using the existing `mma3001` environment (Sphinx/MyST must already be available):

```powershell
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
```

Open `docs/_build/html/index.html` for the build copy or `docs/html/index.html` for the submission copy. See [documentation instructions](docs/README.md) for details. No package installation, training or inference is part of these commands.

## Licence

[MIT License](LICENSE), copyright 2026 Chan Lik Wai, applies only to original project code owned by the repository author. It does not claim ownership of or grant redistribution rights for university-supplied orchard datasets, Roboflow datasets, third-party model weights, RF-DETR code or pretrained weights, or other externally licensed materials. Those remain subject to their respective terms and permissions. The repository includes the MIT licence linked above. Dataset provenance and third-party redistribution permissions remain unverified.

## Limitations and next steps

Colour filtering can miss non-red, shadowed or occluded apples, merge touching apples, and detect other red objects. RF-DETR can also miss, merge, split or hallucinate objects. Neither method establishes true apple counts without independently verified annotations. Standalone GT dependence and unvalidated predicted-branch deployment prevent a verified autonomous deployment claim; mask geometry requires independent confirmation. RGB-D channel scaling does not establish physical depth calibration or 3D localisation. Sample-level validation may include correlated frames and is used for checkpoint selection. No new performance, hardware measurements or experiments are claimed.

Future work should verify the implemented predicted-branch alignment/export stage, evaluate on held-out human apple annotations, and use calibrated depth or reconstructed geometry for 3D positions. Calibrated 3D localisation and independently verified orchard accuracy remain unimplemented objectives.

## References and acknowledgements

Author: Chan Lik Wai. Prepared for the MMA3001 Numerical Methods and Machine Learning project. The implementation uses the TreeID orchard data supplied for the project and an apple COCO export whose precise provenance/licence should be recorded before redistribution. It uses NumPy, OpenCV, Pillow, SciPy, Matplotlib, PyTorch/torchvision and RF-DETR; documentation uses Sphinx/MyST. ResNet-34, U-Net, ASPP and attention/residual blocks describe the implemented architecture. Verified primary background references for these method families are listed in the [model guide](docs/source/models.md). They do not establish the provenance of a previously unnamed source paper; the student must still acknowledge the materials actually used.


## Portability and submission evidence

`environment.yml` pins Python 3.13.15 and PyTorch 2.13.0+cu130. The RF-DETR
export pins Python 3.11.16, torch 2.14.0+cu130, torchvision 0.29.0+cu130 and
rfdetr 1.9.4. Both include Windows runtime packages (`ucrt`, `vc`); the RF-DETR
export contains the author's absolute `prefix`. These are environment snapshots,
not portable lockfiles. CUDA wheels require a suitable package source/cache and
compatible drivers; neither export configures a CUDA wheel index. The RF-DETR
torch/torchaudio/torchvision combination needs clean-install verification.
No dependency versions were changed by this audit. Create RF-DETR at an explicit
chosen path/name rather than relying on the author's prefix. Resolve platform
packages deliberately on other operating systems. The organiser uses Windows paths.

Local data, historical checkpoints and results exist in this working copy but are
ignored by Git. Clone/ZIP recipients need separately permitted data/weight access
and checksums. The [final audit](docs/final_audit.md) records availability, checks,
known issues and the official rubric assessment. The 2026 brief was found in the
parent MMA3001 course directory; its requirements and all six rubric rows are
assessed in the audit, alongside the additional review requirements supplied here.

## AI assistance and student reflection

Codex assisted with this final source/documentation review, documentation edits,
report prose and verification commands. [AI disclosure](docs/ai_disclosure.md)
separates observable assistance from the student's outstanding account of earlier
ChatGPT/Codex use, approximate contribution, decisions, verification and reflection.
No personal contribution percentages or human verification are inferred here.
