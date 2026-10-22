# Final MMA3001 documentation and rubric audit

Audit date: 23 October 2026 (Australia/Sydney). Starting revision:
`b0fdca4bbf1e5fce915f5927a275aeaf9920afed`, branch `main`. The starting working
tree was clean. No commit, push, training, orchard detector rerun or full optimisation was
performed by this audit. Temporary synthetic comparison files were isolated from
project data/results. This assessment is evidence-based, not a predicted mark.

## Primary standard and scope

The official **MMA3001 2026 Project Brief** was found by broadening the search to
`C:/Users/Chan/Downloads/MMA3001/MMA3001_2026_Project_Brief.pdf`.
SHA-256: `1029cf7c7c2844161af0c3727bbcbf40fac3913575683defe34ee942f9a9550c`.
All 15 pages were read through local `pdftotext -layout` extraction. It was not
copied into the repository. Earlier provisional “brief unavailable” assumptions
were superseded before this final assessment.

Requirements on pages 4–7 cover problem/I/O, computational solution, alternatives,
validation, optimisation/performance and AI reflection. Page 8 requires the written
5–10-page document (references/appendices excluded from the ten-page limit), Moodle
submission, accessible repository link and zipped clone. Page 9 requires an
individual five-minute presentation plus three minutes of questions in Week 14,
with a booked appointment. AI cannot be used to answer presentation/interview
questions. Pages 10–15 contain six rubric rows; all are assessed below. Submission
is assessed separately rather than invented as a seventh official rubric row.

The suggested apple-tree reconstruction project on page 3 outputs 3D skeletons.
This repository implements branch segmentation and 2D apple candidates, not that
complete reconstruction. Whether this narrower scope was agreed with teaching
staff is not established by the checkout. No documentation can establish approval.

## Files and evidence inspected

The recursive inventory covered maintained code, tests, all READMEs, documentation
source/API pages, generated HTML/source copies/assets, environment exports,
licence, ignore rules, local dataset/output layout and recent Git history/diffs.
There were 20 maintained Python files initially: 12 production, six test/support
files and two Sphinx configurations. The audit adds one documentation verifier.
Generated/build caches are not additional maintained production modules.
The complete maintained/generated file-path inventory is in `docs/audit_inventory.txt`.
The second-pass `docs/definition_review.tsv` enumerates all 213 maintained named
definitions individually with source line, qualified name, signature, purpose and
documentation-section presence. It is a coverage ledger, not an automatic proof of
semantic correctness. The verifier additionally checks every non-self argument label
and each production parameter's generated API field.

| Production module | Functions/methods/nested helpers | Classes | Main review focus |
|---|---:|---:|---|
| `src/apple_predictor_predicted_compare.py` | 9 | 0 | Predicted/GT sources, cache encoding, projection, checkpoint, subprocess and output provenance |
| `src/apple_predictor_v1.py` | 16 | 0 | RGB/depth, strict colour cutoffs, morphology, component geometry, overlap, exports/timing |
| `src/apple_segmenter_train.py` | 1 | 0 | Import checks, COCO inputs, 50 epochs/batch 1, external library defaults |
| `src/apple_segmenter_v1.py` | 9 | 0 | GT/predicted input masks, confidence/area/overlap, BGR rendering, instance exports |
| `src/compare_apple_detectors.py` | 10 | 0 | CSV validation, assignment, metrics, reports, coverage, source provenance |
| `src/dataset_organiser.py` | 1 | 0 | Import-time copying, Windows paths, restricted PNG decoder |
| `src/full_grid_search_apple.py` | 13 | 0 | Grid, objectives/ties, partitioning, caching, resume and stale held-out bug |
| `src/preprocess_roboflow_dataset.py` | 2 | 0 | Category merge, negative images, preserved instances, missing-copy records |
| `src/train_baseline.py` | 16 | 2 | Independent pixel MLP, preprocessing, weighted BCE, split/checkpoints |
| `src/train_improved_unet.py` | 23 | 6 | ASPP/attention/residual architecture, augmentation, AdamW/scheduler |
| `src/train_unet.py` | 17 | 2 | Four-channel ResNet-34, decoder, plain BCE/Adam, validation |
| `src/winter_inference.py` | 25 | 7 | Architecture reconstruction, checkpoint loading, RGB-D and synchronised timing |
| **Production total** | **142** | **17** | **159 documented definitions, including nested helpers and methods** |

All test/support files were read: `test/__init__.py`, `helpers.py`, `run_report.py`,
`test_apple_detector.py`, `test_apple_comparison.py`, `test_grid_search.py`, plus
`test/README.md` and `test_report.txt`. There are 43 functions/methods and three
test classes in those Python files; 40 methods are discovered tests. Fixtures and
assertions were checked, including the intentional stale-held-out characterization.

Configuration and top-level review included `README.md`, `src/README.md`,
`data/README.md`, `outputs/README.md`, `docs/README.md`, `environment.yml`,
`rfdetr_environment.yml`, `.gitignore`, `LICENSE`, `docs/requirements-docs.txt`,
`docs/source/conf.py`, `docs/static_config/conf.py` and `docs/html_verification.md`.
All seven original Sphinx guides/index were reviewed, together with the API RST
inventory and generated site. The new predicted-branch guide, AI disclosure and
verification utility were reviewed after writing. The generated HTML was parsed
for local targets/anchors and inspected for signatures and updated prose; no GUI
browser interaction or JavaScript search execution is claimed.

Local historical reports `outputs/apple_detector_comparison/evaluation_report.md`
and `outputs/apple_detector_comparison_predicted/evaluation_report.md` were read,
with their metrics/manifests and detector metadata inspected. Their measured files
were preserved. The RF-DETR `hparams.yaml` is an empty mapping locally, so it does
not establish the complete historical training configuration.

Git inspection covered the latest 20 available log entries (18 existed), latest
three commit statistics, current diffs, branch/ref state and tracked/ignored paths.
The newest commit adds the predicted driver; the preceding commit changes tests,
grid documentation and generated HTML. That explains the missing driver API and
stale “unconnected pipeline” descriptions. Local `main` and cached `origin/main`
initially pointed to the same revision. No other branch was visible; meaningful
development commits exist, but remote reachability/current server state was not
verified. Commit messages are often broad and no additional development branch was
visible, so insightful-commit/branching evidence is limited. History was not changed.

## Changes made and why

| File | Documentation change |
|---|---|
| `README.md` | Corrected pipeline scope, added predicted/control reproduction and outputs, API count, portability/availability, official-brief scope caveat and AI disclosure links |
| `src/README.md` | Added new driver/dependency relationship and exact grid values, objective directions, tie differences, provenance and leakage limits |
| `data/README.md` | Explained optional GT, checkpoint/cache requirements and season/path independence |
| `outputs/README.md` | Added predicted/control directories, timing columns/boundaries, cache metadata limits and historical-report caveat |
| `test/README.md` | Explicitly identified missing dedicated driver tests and separated temporary smoke checks from the suite |
| `docs/README.md` | Corrected 12-module coverage and linked new guide, audit and AI disclosure |
| `docs/source/overview.md` | Replaced incorrect unconnected-pipeline claim with actual driver behaviour |
| `docs/source/workflows.md` | Corrected workflow summary and added driver API navigation |
| `docs/source/data_and_preprocessing.md` | Distinguished exported RF-DETR masks from the driver’s inferred/cached masks and GT control |
| `docs/source/models.md` | Added engineering rationale and qualitative trade-offs without unsupported performance claims |
| `docs/source/reviewer_start.md` (new) | Short marker route, reproduction levels, claim boundaries and final submission/presentation preparation |
| `docs/source/training_and_evaluation.md` | Removed obsolete claims about deleted tests and output fixtures; documented actual synthetic coverage |
| `docs/source/usage.md` | Added supported summer predicted/control commands and prerequisites |
| `docs/source/index.rst` | Corrected introduction and indexed the new guide |
| `docs/source/predicted_branches.md` (new) | Complete driver CLI, geometry, inputs/outputs, failure handling, metrics, cache and evaluation limits |
| `src/apple_predictor_predicted_compare.py` | Rewrote eight incomplete function docstrings, documented nested CSV writer, corrected two comment passages; second pass corrected report provenance and branch-source reproduction prose; numerical workflow unchanged |
| `src/apple_predictor_v1.py` | Corrected helper docstrings that incorrectly excluded predicted branch inputs; preserved standalone GT semantics |
| `src/compare_apple_detectors.py` | Corrected module/main prose and five report-text constants: environment/test commands, removed constant, source-default/GT/custom-path claims and tuning provenance |
| `src/train_improved_unet.py` | Removed four unreachable duplicate documentation strings after returns; real class docstrings retained |
| `src/train_unet.py` | Removed one unreachable duplicate description after a return; retained the actual dataset class docstring |
| `src/train_baseline.py`, `src/train_unet.py`, `src/train_improved_unet.py`, `src/winter_inference.py` | Added module-level output paths/files and overwrite behaviour |
| `test/helpers.py` | Documented the one-pixel fixture’s inputs, inclusive geometry and output |
| Three `test/test_*.py` files | Added specific missing class/fixture/test docstrings; corrected false “no main invocation” summary; assertions/names unchanged |
| `docs/static_config/conf.py` | Documented extractor helpers and added a documented successful-build HTML whitespace hook; still imports no production modules |
| `docs/source/conf.py` | Corrected its example to select the maintained static-API configuration |
| `docs/verify_documentation.py` (new) | Standard-library syntax/docstring/API/local-link checker with explicit scope limits |
| `docs/ai_disclosure.md` (new) | Factual audit-session disclosure and student-owned reflection requirements |
| `test/test_report.txt` | Refreshed from actual complete test execution |
| `test/run_report.py` | Documented import-time execution and recorded actual Python/platform/Conda metadata instead of a hard-coded environment label |
| `docs/html_verification.md` | Replaced stale prior-session claims with current verification evidence |
| `docs/final_audit.md` (new) | This complete official-rubric assessment and remaining-work record |
| `docs/audit_inventory.txt` (new) | Explicit maintained/generated file list and historical-report inputs reviewed |
| `docs/definition_review.tsv` (new) | Per-definition signature/purpose/documentation ledger for all 213 maintained definitions |
| `docs/checkpoint_smoke_report.json` (new) | Actual CPU checkpoint/execution smoke outcomes and checkpoint hashes, separate from the 40-test suite |
| `docs/distribution_smoke_report.json` (new) | Actual source-only export test, documentation check and strict rebuild logs in an isolated temporary directory |
| `src/full_grid_search_apple.py` | Second-pass documentation/comment correction: no-cache reloads full split lists and does not establish reduced memory |
| `docs/source/api/` and `docs/html/` | Regenerated API index/pages, copied guide/API sources, HTML navigation, search/object indexes and assets; added driver page and guide |

Generated files are build products of the maintained configuration and current
docstrings, not separately authored API assertions. Unchanged source/test/environment
files were still reviewed. Build-only audit snapshots/scripts remain ignored under
`docs/_build/`; no datasets or weights were added to the source distribution.

## Documentation completeness

- **12 production modules; 142 functions/methods/helpers and 17 classes examined.**
- **48 pre-existing missing definition docstrings corrected:** one nested production
  writer, 45 test/fixture/class definitions and two static-extractor helpers.
  Eight existing driver function docstrings were substantially completed as well.
- All maintained Python modules and named definitions now have docstrings. Simple
  no-argument test assertions use concise behavioural descriptions rather than
  meaningless Args/Returns boilerplate. Production tensor/geometry/error contracts
  are documented; inherited/dynamic third-party APIs remain out of static scope.
- Two inaccurate inline comment passages in the new driver were corrected. Four
  orphan duplicate strings in the improved trainer and one in the basic trainer
  were removed; actual class docstrings were retained.
- Resolved stale guide/API claims include: no connected driver,
  wrong API count, GT-only helper input, external-mask-only workflow description,
  deleted-test coverage, absent new workflow commands and absent grid-space detail.
  The source README's missing winter arguments and binary-network-output claim
  were also corrected. Portability, provenance and disclosure gaps were documented.
- Five generated-report prose constants were corrected at their generator; metric
  expressions, assignment, thresholds, CSV schemas and output defaults are unchanged.
  A second pass also corrected three report expressions in the predicted wrapper:
  predicted/GT reference provenance paragraphs and the source-selection reproduction
  template. These edits do not change comparison inputs or metric calculations.
- The morphology helper now warns that closing can add non-red/invalid-depth
  foreground after initial pixel selection; final components are not strictly
  depth-valid pixel by pixel.
- **12/12 production modules indexed in HTML.** Current checked site has 25 HTML
  pages. Exact final syntax/docstring/link counts are recorded in
  [HTML verification](html_verification.md) and produced by the verifier.

## Official marking rubric assessment

“Pass” below means repository documentation evidence satisfies the audited aspect,
not a guaranteed assessment grade. Excellent also depends on student understanding,
valid experiments and presentation evidence that cannot be manufactured here.

| Official category / page | Excellent expectations | Evidence and improvements | Remaining gap | Status |
|---|---|---|---|---|
| Engineering problem, inputs and outputs / 10 | Complete professional context, precise problem, importance/use, justified scope, input/output formats/units/domains/meaning | Root motivation and robotic-perception scope; RGB-D/channel/GT contracts; all workflows, outputs and failure handling documented | Confirm agreed 2D scope relative to suggested 3D skeleton project; depth calibration and deployment acceptance criteria absent | Partial |
| GitHub repository structure, commits and README / 11 | Logical professional structure, complete orientation/environment/dependencies, reproducibility, appropriate data handling, branching and/or insightful commits | Coherent src/test/data/outputs/docs; full commands, environment limitations, ignored large files, history inspected; isolated candidate source-only export passed tests/docs/build | Clean installation and external data/weight access not demonstrated; exports are Windows/CUDA snapshots; remote accessibility not checked | Partial |
| Inline comments, docstrings and HTML documentation / 12 | Readable intent comments; consistent full parameter/return/exception/usage docs; complete structured HTML synchronised to code | Missing/incomplete docs corrected; current 12-module static API; strict build and local-link checks; no runtime imports for side-effect modules | No claim to document inherited third-party internals or prove GUI/JavaScript behaviour; student should review technical explanations | Pass for maintained repository documentation |
| Validation and engineering credibility / 13 | Appropriate rigorous design, correct metrics/results, GT/withheld/benchmark evidence, unsupported inputs/failure modes and honest limitations | 40 passing synthetic checks; explicit reference-relative metrics, matching/zero cases, validation-selection reuse, geometry/leakage and failure limits | No independent orchard apple labels or winter accuracy; no dedicated driver suite; historical reference/checkpoint provenance incomplete | Partial |
| Justification of Computational Approach / 14 | Credible baseline, justified methods, quantitative accuracy/runtime/memory/robustness/scalability, evidence-supported improvements and trade-offs | MLP/U-Net alternatives and colour/RF-DETR rationale; measured local CSV snapshots clearly qualified; grid spaces/objectives/partition and timing boundaries documented | In-progress grid, no audited final held-out selection; no controlled runtime/memory/scalability comparison; scope and final design conclusions need student evidence | Partial |
| AI use and critical reflection / 15 | Full disclosure/justification/verification, identified errors, student-driven decisions, understanding and thoughtful reflection | Current Codex session disclosed; claims constrained to actual checks; personal-report requirements explicitly listed | Student must supply complete earlier ChatGPT/Codex use, approximate contribution, retained decisions, real checks/errors and personal learning | Needs attention |
| Submission/demonstration requirements / 8–9 (additional requirements, not a rubric row) | README, HTML, licence, functional main, tests/report, accessible link, Moodle report/zipped clone; booked presentation and questions | Repository artefacts present; source-only software/docs export verified; marker route and supported commands provided | No Moodle/ZIP/link verification, no separate report found, no appointment/presentation evidence, no clean installation/full orchard demonstration | Partial |

## Actual verification and limits

Environment: existing `mma3001`, Python 3.13.15, Windows 11 build 26200;
NumPy 2.5.2, OpenCV 5.0.0, Pillow 12.3.0, SciPy 1.18.1, Matplotlib 3.11.1,
torch 2.13.0+cu130, torchvision 0.28.0+cu130, Sphinx 9.1.0, MyST 5.1.0.
These imports succeeded. Existing `rfdetr` imports also succeeded with Python
3.11.16, torch 2.14.0+cu130, torchvision 0.29.0+cu130 and rfdetr 1.9.4;
CUDA was available, with torch JIT deprecation warnings. No clean installation,
dependency changes or GPU inference benchmark was performed.

```powershell
conda run --no-capture-output -n mma3001 python test/run_report.py
conda run --no-capture-output -n mma3001 python docs/verify_documentation.py
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
git diff --check
```

Full suite: **40 passed, zero failures/errors/skips**; actual output is preserved in
[test report](../test/test_report.txt). The stale-held-out test intentionally passes
when the known bug is reproduced. It is not a regression fix.

Seven production CLI `--help` checks exited zero: three branch trainers, winter
inference, predicted driver, saved comparator and full grid search. Other production
scripts configure through constants or perform import-time work; passing them
`--help` would not be a safe help-only check and was intentionally avoided.
Documented flags/defaults were cross-checked against parser definitions, including
all predicted-driver options and required branch-training/checkpoint arguments.
Help success does not establish training/inference success.

Isolated smoke checks passed for central projection/padding and agreement with
RF-DETR geometry for intended square masks; zero/255 cache acceptance and
maximum-one rejection; and complete zero-detection comparator output/report
generation in temporary directories. The second pass extended these checks:

- All three local branch checkpoints loaded strictly on CPU with their matching
  inference architectures. MLP/basic U-Net produced finite `[1,1,256,256]` logits
  for zero tensors; improved U-Net processed a temporary synthetic RGB-D frame.
- The predicted driver exported a fresh 256-square binary mask and full-frame
  detections, then successfully reused the cached mask. Its GT control exported
  the expected one red component with the correct source/alignment metadata.
- Both wrapper comparison/report paths ran against temporary synthetic reference
  CSVs, checking source provenance wording and the GT branch-source command.
  The synthetic reference was copied from the control solely to verify software
  consistency; perfect self-agreement is not accuracy evidence.

All temporary images/results were isolated from `data/` and `outputs/`. These
trained-weight smoke checks establish local loading/execution compatibility only.
Their outcomes and checkpoint hashes are preserved in `docs/checkpoint_smoke_report.json`.
No orchard inference, training, full optimisation or real-data accuracy test was
performed. No RF-DETR model was run.

A further source-only distribution check copied the candidate working files
(including uncommitted documentation) selected by Git's tracked/unignored inventory
into a temporary directory. Both `data/` and `outputs/` contained only their README;
Git metadata, ignored images, checkpoints and measured results were absent. All
40 tests, the documentation verifier and a strict HTML rebuild passed there using
the existing `mma3001` interpreter. Complete command output is retained in
`docs/distribution_smoke_report.json`. This verifies software/documentation
independence from local ignored inputs. It is not a clean environment installation,
remote clone/access check or full orchard experiment reproduction.

Primary background references for ResNet, U-Net, ASPP, attention gates and the
official RF-DETR software were checked and added to the model guide. They identify
method families, not undocumented historical student research provenance; paper
performance is not used as local experimental evidence.

Python compilation is in memory for all 21 maintained Python files, including the
verifier, without executing preparation/training side effects. Local HTML href/src
targets and fragments are checked; inline Markdown file links are checked. External
URL availability, CSS url() references, Markdown reference-style links/heading
anchors, GUI layout and interactive search are outside that automated check.

Source preservation was checked against a pre-edit snapshot: production modules and
test/fixture ASTs match after excluding documentation strings, the five comparator
report-prose constants and three predicted-wrapper report expressions. Documentation infrastructure
intentionally adds the Sphinx formatting hook and actual test-report environment
metadata. Production algorithms and test assertions are preserved. No architecture,
optimiser, split, numeric threshold,
matching/metric expression, CSV schema or default result path was changed.

The pre-edit snapshot covered **18,137 data/output/environment files** with SHA-256.
Only `outputs/apple_grid_search/all_experiments.csv` changed during audit checks;
no audit command wrote it. An external writer appears active. All other snapshotted
protected files matched. Consequently this audit cannot certify globally unchanged
results while that external process runs. The CSV was preserved as found and its
partial row count is time-dependent.

## Local availability and measured snapshots

At inspection the organised layout contained 562 summer RGB, depth and GT PNGs,
562 winter RGB/depth PNGs and zero winter GT PNGs. Counts do not verify registration,
annotation correctness or independence. Current default MLP, basic U-Net, improved
U-Net and RF-DETR checkpoint paths were present locally. They are ignored by Git.
The predicted-run metadata points to historical
`outputs/improved_resnet34_unet/improved_resnet34_unet_best.pth`, which was absent,
whereas the current `improved_resnet34_unet_output` checkpoint was present. Do not
assume these are identical: inspect hashes before claiming exact reproduction.

Branch `val_metrics.csv` files and the winter timing summary also contain historical
results. They were inspected without rerunning models:

| Model | Saved summer validation Dice | Saved validation IoU | Saved winter mean forward ms |
|---|---:|---:|---:|
| Pixel MLP | 0.186075 | 0.102581 | 2.198292 |
| Basic U-Net | 0.331765 | 0.198872 | 5.845189 |
| Improved U-Net | 0.539385 | 0.369286 | 8.378011 |

The validation files are under each current default training directory; winter
values come from `outputs/winter_inference_benchmark/summary.csv`. Winter metadata
records 562 images, input side 256, threshold 0.5, five warmups, CUDA and an NVIDIA
GeForce RTX 5060 Laptop GPU. That is historical metadata, not newly measured
hardware evidence. Its basic/improved checkpoint paths use historical directories;
identity with current checkpoints has not been established. Summer validation and
winter timings are different populations, and winter has no GT. These numbers
suggest a saved accuracy/computation trade-off but do not certify current-code
reproduction, independent test performance, end-to-end latency or scalability.

Existing apple summary CSVs contain the following **historical saved agreement results**;
they were read, not recomputed or certified as independent accuracy:

| Saved experiment | Frames | Precision | Recall | F1 | Count MAE | Count RMSE | Mean matched distance px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone GT/full resize | 562 | 0.856258 | 0.393047 | 0.538779 | 17.683274 | 20.059697 | 4.496633 |
| Predicted/central placement | 562 | 0.963349 | 0.439676 | 0.603783 | 17.708185 | 20.009517 | 4.202143 |

Source files: the respective `outputs/apple_detector_comparison*/summary_metrics.csv`.
Both use 30-pixel tolerance and 18,293 saved reference detections. These rows do not
isolate branch-source quality because geometry differs; an aligned GT control is
needed for that comparison. Increased reference F1 does not prove increased true
apple accuracy. No aligned-GT result folder was available at inspection. Grid CSV
contained 2,726 rows when read (of 5,184); it was being modified externally, so no
final optimum or held-out result is claimed. No selection objective was changed.

## Remaining issues by severity

### Critical

No newly verified critical computational failure was found in the checks performed.
This is not certification of full training/inference/deployment functionality.

### High

1. **Executable/provenance — `src/full_grid_search_apple.py:main`:** resume metadata
   omits `--select-by`; an existing `heldout_comparison.csv` suppresses reevaluation.
   The actual characterization test reproduces this. Use a fresh output directory
   per objective. Separate fix: record objective/selected parameters and invalidate
   or validate held-out output on resume, with a real regression test.
2. **Executable/provenance — `apple_predictor_predicted_compare.py:detect_and_export`:**
   reused masks are not fingerprinted or checked against checkpoint/input size/
   threshold, although metadata records requested values. Separate fix: persist and
   validate per-mask/run provenance before reuse. For now use fresh mask folders.
3. **Validation/scope:** independent apple GT, winter quantitative accuracy, agreed
   2D scope, calibrated depth/3D outputs and deployment acceptance evidence are
   absent. Benchmark agreement alone cannot establish those claims. The brief does
   allow credible benchmark validation, but conclusions must stay within its limits.
4. **Reproducibility/submission:** a clone lacks datasets/weights/most results.
   Resolve permitted access, hashes, provenance and a clean-clone trial. Stop or
   finish externally running optimisation before freezing final evidence; do not
   present partial rankings as exhaustive.

### Medium

1. **Validation — training `main` functions:** seeded sample splits may mix correlated
   frames across train/validation; final validation is checkpoint-selection data.
   Predicted summer inference may reuse these frames. Separate experimental work
   needs held-out trees/human annotations and recorded partition identities.
2. **Executable/report semantics — predicted driver `run_comparison`:** paragraph
   replacements remain prefix-dependent. The second pass corrected unsupported GT
   reference assumptions and the GT-source reproduction command, but there is still
   no automatic reference provenance validation or complete invocation replay.
   A separate design should generate reports from explicit validated experiment metadata.
3. **Environment:** Windows runtimes, CUDA-tagged wheels with no configured index,
   an absolute RF-DETR prefix and untested clean installation limit portability.
   Existing imports succeed; that does not certify the exported wheel combination
   or installation elsewhere. Dependency versions were preserved.
4. **Executable/unsupported input — trainers `main`, grid `score`:** CLI numeric
   ranges are incompletely checked; zero training epochs can fail on `history[0]`;
   `count_bias` can return NaN instead of ranking it last. Code inspection verifies
   these paths; no separate bug-fix regression is claimed. Use valid inputs now;
   recommend distinct validation fixes/tests.
5. **Executable side effects — organiser and COCO preparation:** importing organiser
   copies files; preparation/training import checks require real paths. Missing
   source-image copies leave COCO records. Documented; not changed under this task.
6. **Written report/presentation:** student-owned method-selection justification,
   engineering interpretation, quantitative performance trade-offs, lessons,
   references and complete AI reflection remain outstanding evidence. No personal
   reflection or contribution percentage has been fabricated.
7. **Executable/performance — `full_grid_search_apple.py:main` and nested
   `get_frames`:** `--no-cache` uses a full list comprehension for each split;
   it does not stream images. The previous `current_frames` can remain alive while
   the next full list is built. Source inspection verifies the allocation/lifetime
   pattern; no peak-RAM benchmark was performed. CLI help's intended RAM reduction
   is unsupported. Docs now state the behaviour. Recommend a separate streaming or
   explicit lifetime-management change, followed by memory measurements.

### Low

1. **Executable console prose — `train_improved_unet.py:main`:** prints “Optimiser:
   Adam” while constructing AdamW. Accurate docs state AdamW; executable print text
   was preserved under the no-computational-edit scope. Correct separately.
2. **Historical documentation:** ignored generated reports still contain historical
   prose/commands, and a metadata checkpoint path is no longer present. Results
   were preserved; future generator text is corrected and caveats added.
3. **Repository packaging:** final `outputs/` ignore rule prevents adding new files
   beneath it despite the earlier README exception. Existing tracked
   `outputs/README.md` remains included. No new result inclusion is implied.
4. **Presentation/rendering:** graphical layout and JavaScript search were not
   exercised. Open the site offline with assets together before submission.

## Completion and personal submission checks

The maintained documentation now covers all production modules and current
workflows, including the predicted/GT control distinction, CLI/defaults, geometry,
units, logits/scores, metrics, output contracts, optimisation limitations, accurate
test scope and AI-session disclosure. API pages were regenerated from current
source; feasible syntax, software, help, import and local-documentation checks were
performed. Computation and measured files were not intentionally modified.

An Excellent-level submission remains **partially evidenced**, principally because
documentation cannot supply independent validation, complete run provenance,
clean-clone reproducibility, a finished optimisation, the student’s personal
reflection or presentation performance.

Before submission, the student must:

1. Confirm project scope with teaching staff and compare the actual written report
   against the official brief, including meaningful engineering interpretation.
2. Complete the 5–10-page report and truthful AI contribution/reflection; supply
   dataset/method references and resolve redistribution permissions.
3. Resolve or explicitly retain the code/provenance limitations above, freeze a
   completed experiment revision and avoid unsupported independent-accuracy claims.
4. Demonstrate a fresh clone with permitted external inputs/checkpoints and review
   the actual generated outputs; verify HTML offline and final GitHub access.
5. Commit/push the reviewed documentation, verify the final main branch, submit the
   accessible link and zipped clone plus report via Moodle, and book/attend the
   five-minute presentation with three minutes of unaided questions. Due dates are
   in the separate Unit Overview Spreadsheet, not established by this checkout.
