# Documentation verification

Verification date: 21 October 2026. Scope: documentation-only update for the MMA3001 repository. No source revision was committed, and no packages were installed, files downloaded, training/inference run, or generated data/results changed.

## Baseline and inventory

The working tree was clean at the initial `git status --short`. Tracked-file inventory and filesystem inspection identified 12 maintained Python modules (including tests and Sphinx configuration), five maintained READMEs, five existing Markdown guides, the Sphinx index/configuration and documentation requirements. The index and dependency specification were read but not changed. The data/output folder READMEs are explicitly maintained, even though their neighbouring contents are excluded.

Original text for all 22 files selected for editing was captured before edits in a temporary JSON snapshot outside the repository. It includes original docstrings and the UTF-8 BOM where present. Python parsing accounts for that valid encoding marker; it is not a syntax defect. Virtual environments, datasets, model files, third-party files, caches, generated HTML, results, figures and generated evaluation reports were excluded. There are no additional tracked Python/Markdown files outside this inventory.

## Coverage and checks

Final coverage: **12 modules, 18 classes and 129 named functions/methods**, including constructors, dunder methods and both nested helpers. Every item has a docstring. All non-self/cls named parameters appear in Args sections. Google-style Returns, Yields and Raises sections describe relevant supported behaviour; examples are included where useful. Anonymous lambdas are unchanged executable expressions, not named function definitions.

- Syntax: all 12 files compiled in memory successfully; no project modules were imported and no bytecode cache was written.
- AST: all 12 match the baseline after removing only leading string-expression docstring nodes in modules/classes/functions. Comparisons ignore AST location attributes only; non-docstring string expressions, constants, imports, signatures and statement ordering remain included.
- Tokens: after removing only docstring tokens, comments and whitespace/indentation/newline tokens, every remaining token type and value is identical to baseline. One-line suites were split using whitespace solely to insert docstrings; their executable tokens are unchanged.
- Coverage: no missing module/class/function docstrings or named-parameter descriptions were found.
- Markdown: relative local links and direct Python-script commands resolve. Script command options were checked against argparse source declarations. Default output paths/checkpoint names were checked against source. Data/checkpoint paths and generated HTML/output files are explicitly labelled requirements or expected products, not guaranteed committed files. External bibliography, software compatibility and dataset provenance were not independently verified.
- Git: `git diff --check` passes. The full diff was reviewed by file; allowed changes are Python comments/docstrings/whitespace and Markdown. No imports, dependencies, parameters, algorithms, signatures, executable string data, data formats or directory names changed.

All preservation checks cover the entire maintained Python inventory, not a selected subset. No executable discrepancy required repair. Updating a module docstring can change displayed help text where argparse reads `__doc__`; this is documentation text, not changed command parsing or algorithm behaviour.

No runtime test suite or HTML build was run. The user prohibited training/inference, the full tests include all-frame detection, and static checks satisfy the documentation preservation requirements without executing application code. Synthetic test commands are documented for separately intended runs. No test-pass or benchmark-performance claim is made.

## Per-module inventory

The final column fingerprints the stripped AST representation used in the preservation comparison (SHA-256), not source-file bytes or model checkpoints. Each row passed syntax, AST equality and executable-token equality.

| Python module | Classes | Named functions/methods | Stripped AST SHA-256 |
|---|---:|---:|---|
| [src/apple_predictor_v1.py](../src/apple_predictor_v1.py) | 0 | 16 | `0b806a0f1d3cc02802b9a83f2ce0c5ea97d2dd7ca0e6ea216bf17128ea14ae1a` |
| [src/apple_segmenter_train.py](../src/apple_segmenter_train.py) | 0 | 1 | `78efda74f94055e37e6fa3cd86cf4fa2f80fe7bd567e759583ce103b121882f4` |
| [src/apple_segmenter_v1.py](../src/apple_segmenter_v1.py) | 0 | 9 | `fdc62dfafa1da80563f15c2a27d8cd506b0bb20964d08ff5bcf71f33bf121afd` |
| [src/compare_apple_detectors.py](../src/compare_apple_detectors.py) | 0 | 10 | `f7e117e5574e79f2dde8788add7b0a9416604cce1d7a340f0d0e2270391aaa4c` |
| [src/dataset_organiser.py](../src/dataset_organiser.py) | 0 | 1 | `450d52108a217dba6ab3b46da134537086d4aeda0aae2b94a2c0e73490b4a9a3` |
| [src/preprocess_roboflow_dataset.py](../src/preprocess_roboflow_dataset.py) | 0 | 2 | `30a4c564bc5c10a866c5720f9fb6d25092d1269a8d88b2c8143effb20c38cde6` |
| [src/train_baseline.py](../src/train_baseline.py) | 2 | 16 | `1706a24a69032b8621e7fe5902270bd2518fd8ced95840722295f16751e592d2` |
| [src/train_improved_unet.py](../src/train_improved_unet.py) | 6 | 23 | `7abc8bd3e7e8560798fed48d6eb7ccf2ad9e926cdd99097534aafb1d999abb61` |
| [src/train_unet.py](../src/train_unet.py) | 2 | 17 | `0f8e49872d24e4e126a7d6f2dd505eb48af5804ba29a4851ceb6add4a764801e` |
| [test/test_apple_evaluation.py](../test/test_apple_evaluation.py) | 1 | 9 | `71a744a288116cbab2f021115b7a51086815aac63f3bde850f7bb6276a3ae1a4` |
| [test/winter_inference.py](../test/winter_inference.py) | 7 | 25 | `e5dee1c305afa5d7787fbe4dce02f7afc0242a1e5e9cfa9d6274c2633eb7ab57` |
| [docs/source/conf.py](../docs/source/conf.py) | 0 | 0 | `950ef3e3a75a72067d59cc2c3cc0bd960670389ba78d425c4c3aa0b4a5e76529` |

## All files changed

The 12 Python modules above were documented. All five READMEs were updated:

- [README.md](../README.md)
- [src/README.md](../src/README.md)
- [data/README.md](../data/README.md)
- [outputs/README.md](../outputs/README.md)
- [docs/README.md](README.md)

All five existing Markdown guides were reviewed and updated:

- [Overview](source/overview.md)
- [Data and preprocessing](source/data_and_preprocessing.md)
- [Model architectures](source/models.md)
- [Training and evaluation](source/training_and_evaluation.md)
- [Usage](source/usage.md)

This verification report is the only new repository file. Total: 23 files (22 existing files plus this report).

## Pre-existing errors and limitations identified by inspection

1. `test_export_coordinate_centroid_and_csv` patches nonexistent `detector.CSV_PATH`; patching raises AttributeError with default create=False. The real writer uses COUNTS_CSV, INSTANCES_CSV and TIMINGS_CSV as separate constants. Patching only OUTPUT_ROOT does not redirect those previously computed constants. Its result fixture also omits read_ms, detection_ms, save_ms and total_ms required by save_csv. No fix or export attempt was made.
2. `test_actual_instance_count_consistency` expects `outputs/apple_rfdetr_boxes_only`, while the current inference default is `outputs/apple_rfdetr_instances`. The expected historical directory was absent in the inspected top-level output listing. This test requires external saved fixtures and is not universally runnable from a clean checkout.
3. The regression test runs detection for all complete frames and reads the current Git HEAD source, so it requires dataset/history and is not an independent fixed golden baseline or an inference-free test.
4. Branch trainers accept zero/negative numeric options without comprehensive validation. Zero epochs leave history empty and can fail at history[0] during CSV export. This is documented, not corrected.
5. The organiser performs copying at import; COCO preprocessing and RF-DETR training perform dataset checks on import. The minimal PNG reader does not validate signature, CRC or interlace. Missing COCO images remain as records after copy warnings.
6. The two U-Net training files contain misplaced non-leading string expressions inherited from earlier documentation. Proper class docstrings were added, but those expressions were retained because removing them would alter the executable AST under the user's comparison rule.
7. Comparison report text is embedded executable string data. It contains a stale MIN_RED_AREA remark and statements about historical tuning/reruns that cannot be established by source inspection. It remains unchanged, as do existing generated reports/results. Maintained Markdown explains this limitation.

All original Python files were syntactically valid. Runtime issues above were identified statically, not reproduced by executing failing tests, training or inference.

## Unsupported or stale documentation corrected

- Replaced nonexistent script names (including apple_predictor.py, train_pixel_mlp_documented.py, resnet34_unet_train.py and older crop_trainval variants) with actual maintained filenames.
- Corrected traditional annotations from an annotated/ subfolder to flat outputs/apple_detection; corrected current RF-DETR output to apple_rfdetr_instances and documented current branch checkpoint defaults separately from historical folders.
- Replaced unsupported tree3d environment assumptions with the current treeid workflow examples, explicitly stating that no runtime lockfile/environment YAML is supplied and compatibility is unverified.
- Removed example console counts/averages and historical dataset-count/dimension tables as unverified observations. Geometric dimensions remain only as conditional preprocessing examples.
- Corrected extended training from Adam to AdamW with its actual scheduler/augmentation; intermediate features are no longer described as one-channel branch logits.
- Removed unnamed-paper implementation claims that could not be sourced. Architectural descriptions and the missing bibliographic attribution are explicit.
- Clarified that branch GT is not apple annotation; RF-DETR is a pseudo-reference; default apple workflows use GT, not an automatically connected branch prediction pipeline.
- Corrected full-frame versus central-square mask alignment, >0 versus >127 cutoffs, 0/255 predicted-mask encoding and depth usage. Preserved the implementations and warned against interpreting the comparison as a controlled detector-only experiment.
- Replaced implied guaranteed speed/accuracy rankings with architectural intent. Clarified that HTML is built locally, not already available as a hosted/committed site.

## Unresolved documentation gaps

The repository does not supply a complete runtime environment specification, supported dependency/version matrix, minimum hardware requirements, independently verified GT field of view, calibrated depth units, dataset licence/provenance, full paper citation, independently verified held-out orchard apple annotations or certified provenance for historical checkpoints/CSVs. These facts were not invented. A fully reproduced experiment, runtime tests and a built HTML-site rendering remain unverified. The existing Sphinx site remains branch-focused; the maintained READMEs cover the apple workflows and the Python source holds the API contracts.
