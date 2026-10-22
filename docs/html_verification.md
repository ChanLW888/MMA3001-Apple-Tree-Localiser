# Documentation verification, 23 October 2026

The [final audit](final_audit.md) records the official brief/rubric, files reviewed,
changes, measured snapshots and unresolved issues. This supersedes the previous
session's verification report. Results below are actual checks in this checkout.

| Check | Result |
|---|---|
| Maintained Python compilation, in memory | 21/21 passed without executing modules |
| Production inventory | 12 modules, 142 functions/methods/helpers, 17 classes |
| All maintained definitions, tests/config/verifier included | 213 definitions; no missing module/definition docstrings |
| Existing missing definition docstrings corrected | 48; eight existing driver docstrings also substantially completed |
| Synthetic suite | 40 passed; zero failures/errors/skips |
| CLI help | Seven argparse entry points exited zero |
| Sphinx | 9.1.0; strict builds of both build/submission copies succeeded without warnings |
| HTML API | 12/12 modules indexed; all 159 production definition anchors present |
| Parameter documentation | All 286 non-self argument labels present; all 273 production parameter fields generated |
| Generated HTML | 25 pages; 1,313 local href/src references and fragments resolved |
| Maintained Markdown | 17 files; 73 local inline file links resolved |
| Whitespace | git diff --check passed after template whitespace normalisation |
| Temporary smoke checks | Central projection/padding, cache encoding, zero-detection comparator/report generation passed |

Commands from the repository root:

```powershell
conda run --no-capture-output -n mma3001 python test/run_report.py
conda run --no-capture-output -n mma3001 python docs/verify_documentation.py
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
git diff --check
```

The verifier prints current counts and failure lists; future additions change counts.
It checks local inline Markdown file links and HTML href/src targets/fragments.
External URLs, reference-style Markdown/heading anchors, CSS url() and GUI/JavaScript
behaviour are outside its scope. Static extraction excludes inherited/dynamic
third-party APIs. Actual test output, timestamp, interpreter, Python, platform,
Conda environment and exit code are in [test report](../test/test_report.txt).

The first strict build failed on an unescaped wildcard; it was corrected and both
final strict builds passed. Generated template whitespace was corrected through a
documented Sphinx build-finished hook. No packages were installed. Existing
mma3001 scientific/documentation imports and rfdetr torch/torchvision/RF-DETR imports
succeeded. RF-DETR emitted torch JIT deprecation warnings. No clean install was run.
One concurrent Conda launch encountered a temporary-file lock; both final builds
were rerun sequentially and exited zero.

Original production/test ASTs match after stripping documentation strings and the
five authorised comparator report-prose constants and three predicted-wrapper
report expressions. Algorithms/assertions match.
Documentation infrastructure intentionally adds the Sphinx formatting hook, verifier
and actual test-report environment metadata. Numerical paths and both YAMLs remain
unchanged.

18,137 data/output/environment files were snapshotted. One grid-results CSV changed
without an audit write to it, consistent with an external writer. All other protected
hashes matched. Historical reports/results, checkpoints and data were preserved;
no global unchanged-results guarantee is made while that process continues.

Second-pass CPU smoke checks strictly loaded all three real local branch checkpoints;
MLP/basic U-Net produced finite logits on zero tensors, while improved U-Net ran on
a temporary synthetic RGB-D frame. Fresh/cached predicted exports, GT control,
metadata and both comparison/report paths succeeded using temporary synthetic
reference CSVs. All 213 named definitions are listed in `definition_review.tsv`.
The 40-test suite count is unchanged; these are separate smoke checks.

These checks do not establish orchard/model accuracy, full-dataset model execution,
GPU timing, clean-clone reproducibility or complete optimisation. The known stale
held-out reuse is reproduced, not fixed. See the final audit for remaining issues.

A candidate source-only export (including uncommitted documentation, with ignored
orchard files/checkpoints/results and Git metadata absent) passed all 40 tests, the
documentation verifier and a strict HTML rebuild in a temporary directory. This
used the existing environment; it does not certify clean installation or remote
repository access. Raw output is retained in `distribution_smoke_report.json`.
