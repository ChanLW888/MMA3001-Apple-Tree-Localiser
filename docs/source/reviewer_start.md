# Reviewer starting point and reproduction levels

This project studies RGB-D branch segmentation and 2D apple localisation as a
perception stage for later orchard engineering work. It does not implement a
calibrated 3D tree skeleton, robot controller or verified autonomous harvesting
system. Confirm the agreed scope against the project brief.

## Review without orchard data or checkpoints

Start with [methods](models.md), [input contracts](data_and_preprocessing.md),
[validation](training_and_evaluation.md), [apple workflows](workflows.md) and the
[complete API index](api/index.rst). The API is extracted statically, including
methods and nested helpers, so viewing/building it does not run preparation or
training. Keep the entire HTML directory together when opening it offline.

From the repository root, using the documented environment:

```powershell
conda activate mma3001
python -m unittest discover -s test -p "test_*.py" -v
python docs/verify_documentation.py
python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
```

The first command checks synthetic detector, matching, CSV and grid behaviour.
The second checks syntax, definition/argument documentation, generated API coverage
and local documentation targets. The third rebuilds HTML with warnings treated as
errors. None requires orchard data, model weights, a GPU or RF-DETR inference.
Sphinx/MyST and scientific test dependencies must already be installed; the
documentation verifier itself uses only the Python standard library.

Passing software checks establish the tested implementation behaviour. They do
not establish fruit accuracy, tree reconstruction quality or generalisation.
The grid suite deliberately reproduces a known stale-held-out-result bug; that
passing characterization is not a fix.

## Reproduce models and orchard experiments

Obtain the permitted orchard PNGs, cleaned COCO training data and/or required model
weights separately. They are ignored by Git and are not included in a normal clone.
The repository root README and data/output READMEs describe exact paths and schemas.
An exported Conda environment is a Windows/CUDA snapshot, not proof of clean-machine
installation. Start from the required Python versions and verify installed imports
and wheel/driver compatibility before running expensive workflows.

Use [branch training and checkpoint selection](training_and_evaluation.md),
[winter/branch commands](usage.md) and the [predicted/GT experiment guide](predicted_branches.md).
Use separate directories for each run and retain the full invocation, source
revision, package versions, device, input/checkpoint hashes and branch geometry.
Stop before overwriting any historical experiment folder.

The four apple experiment meanings are distinct:

| Experiment | Interpretation |
|---|---|
| Standalone traditional detector | Uses GT >0 stretched across the full RGB frame |
| Predicted-branch driver | Uses improved-U-Net sigmoid masks projected centrally |
| Optional aligned-GT control | Uses GT >127 with the driver's central projection |
| Saved RF-DETR comparison | Uses model predictions as a pseudo-reference, not independently verified apple GT |

Comparison precision/recall/F1 measure reference agreement. Matched-centroid error
and box IoU are conditional on accepted matches. Summer validation selects model
checkpoints and is not an independent test; winter predictions have no quantitative
GT evaluation. Historical CSV results must be traced to their actual input geometry
and checkpoint, rather than assumed to match current source defaults.

Grid search varies seven traditional parameters over 5,184 combinations, using
development trees by default. Only a complete search evaluates the selected row
and baseline on held-out trees. Use a separate directory per objective because
changed objectives can reuse stale held-out results. `--no-cache` reloads whole
split lists; it does not implement bounded-memory streaming. Neither flag removes
the requirement for real inputs and saved reference CSVs.

## Evidence for assessment and final demonstration

The root README links the actual software-test report, definition
ledger, checkpoint smoke report and AI disclosure. The audit maps every official
rubric row to evidence and remaining gaps. Documentation coverage and successful
synthetic checks cannot supply missing independent validation or student reflection.

For the five-minute presentation, demonstrate the engineering problem and primary
I/O, repository/HTML navigation, one safe verification command, the method-selection
evidence, a principal limitation and truthful AI use. Prepare to explain matching,
preprocessing, validation leakage and known defects unaided during the three-minute
question period. This is a preparation outline, not a script claiming personal
understanding or teaching-team approval.

Before final submission, personally verify the reviewed revision is committed and
pushed, the repository is accessible to the marker, the submitted clone includes
the HTML assets and reports, and external input access is explained. Submit the
5–10-page report, accessible link and zipped clone through the required Moodle
channels, and confirm the presentation appointment. References/appendices are
excluded from the brief's ten-page maximum. Check dates in the Unit Overview
Spreadsheet. Do not use a working-folder ZIP containing private or unlicensed
ignored datasets as a substitute for the required zipped clone.
