# Synthetic software verification

Run from the repository root after installing [requirements.txt](../requirements.txt) using the [root setup instructions](../README.md#environment-and-hardware):

```powershell
conda activate mma3001
python -m unittest discover -s test -p "test_*.py" -v
```

Uses built-in `unittest` and the project's existing NumPy, OpenCV, Pillow, SciPy
and Matplotlib dependencies. No dependencies are installed by the suite.
[test_report.txt](test_report.txt) contains captured execution output and limitations.

| Module | Verification |
|---|---|
| `test_apple_detector.py` | Red/depth masks, opening/closing, component area boundary, box/centroid geometry, branch filtering, shape validation, zero detections and temporary CSV export |
| `test_apple_comparison.py` | Inclusive matching tolerance, one-to-one maximum cardinality, hand-calculated metrics/count errors/IoU, exact names, empty detections, CSV validation and duplicate handling |
| `test_grid_search.py` | Actual Cartesian grid, baseline, fixed branch parameters, objective directions, tree grouping, CSV persistence, synthetic evaluation and mocked resume orchestration |
| `helpers.py` | Shared fixtures and imports of only the three audited safe modules |

Tests use small arrays, temporary directories and automatically restored patches.
They do not read orchard images, checkpoints or saved experimental predictions;
they do not write to `data/` or `outputs/`. No GPU, network or RF-DETR import is
needed. The mocked orchestration test invokes the real grid `main` with one tiny
configuration, synthetic modules and a temporary output directory. It never
performs a real optimisation. Grid enumeration does not evaluate configurations.

The organiser performs copying on import. COCO preparation and RF-DETR training
check real paths on import. Those modules and all model workflows are excluded
from runtime imports; Sphinx inspects them statically instead.

## Known issues and limits

`test_changed_objective_reuses_heldout_known_limitation` reproduces an existing
limitation in `full_grid_search_apple.main`: resume metadata omits `--select-by`,
and an existing held-out CSV suppresses reevaluation. The test deliberately
asserts the observed stale reuse; a passing status does **not** certify safe
objective changes. Use a separate output directory per objective. No source fix
was authorised. Source docstring changes also invalidate earlier resume hashes.
Image bytes, environment versions and full test-name provenance are not recorded
in grid metadata. `score(..., 'count_bias')` does not reject NaN as `numeric` does.

This is software unit testing, not validation of apple accuracy, trained weights,
training convergence, dataset alignment, calibration, generalisation, GPU execution
or reproducibility of historical experiments. No all-frame or Git-HEAD regression
oracle is used. Synthetic fixtures cannot establish performance on real orchards.
No full grid search, preparation, training or RF-DETR inference is executed.
The default grid orchestration is tested only with mocked I/O, not real data.

The new `apple_predictor_predicted_compare.py` driver has no dedicated test module
in this suite. The final audit records separate temporary smoke checks for central
projection, cached-mask encoding and report generation. Those checks do not verify
trained branch predictions or replace real held-out annotations.

Every test and fixture now describes its specific expected behaviour in a docstring;
test names, assertions and computational logic are unchanged by the audit.

To refresh the captured report with the same discovery command, run
`conda run -n mma3001 python test/run_report.py`. It preserves actual combined
stdout/stderr and the process exit code, and exits with that same status.
