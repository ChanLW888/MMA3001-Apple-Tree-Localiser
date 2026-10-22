# Final documentation and preservation verification

Verified against a byte-hash/source snapshot captured from the working tree before edits.
The pre-existing deletion of `src/optimise_apple_predictor.py` was preserved.

## Source audit

| Module | Functions/classes | Missing docstrings | AST unchanged |
|---|---:|---:|---|
| `src/apple_predictor_v1.py` | 16 | 0 | Yes |
| `src/apple_segmenter_train.py` | 1 | 0 | Yes |
| `src/apple_segmenter_v1.py` | 9 | 0 | Yes |
| `src/compare_apple_detectors.py` | 10 | 0 | Yes |
| `src/dataset_organiser.py` | 1 | 0 | Yes |
| `src/full_grid_search_apple.py` | 13 | 0 | Yes |
| `src/preprocess_roboflow_dataset.py` | 2 | 0 | Yes |
| `src/train_baseline.py` | 18 | 0 | Yes |
| `src/train_improved_unet.py` | 29 | 0 | Yes |
| `src/train_unet.py` | 19 | 0 | Yes |
| `src/winter_inference.py` | 32 | 0 | Yes |

All 11 source modules compiled in memory without importing them. Executable ASTs
and non-docstring/non-comment tokens match the initial working tree.

## Protected files

16991 snapshotted data/output/environment files retain their original SHA-256 hashes.
Only `outputs/README.md` was intentionally edited under outputs. Neither Conda YAML was changed.
Externally observed changes during this session: ['outputs\\apple_grid_search\\all_experiments.csv'].
No task command wrote experimental results; changed files were left untouched.
Cannot certify unchanged experimental results while external writes occur.

## HTML verification

22 HTML pages checked; 1105 local links/assets/anchors resolved.
Every maintained source module has an API page with documented signatures.
Navigation and required assets are local and available for offline file opening.
No graphical browser or interactive search behaviour was tested.
Sphinx 9.1.0 strict rebuild (`-E -a -W --keep-going`) succeeded without warnings.

## README links

Checked 34 local README file links.
Unresolved README links: []

## Software verification

40 tests passed; 0 failed, 0 errored, 0 skipped in the existing mma3001 environment.
See [actual execution output](../test/test_report.txt). No packages installed or changed.
This does not measure real-world model accuracy. The known stale held-out reuse
in full_grid_search_apple.main is reproduced and documented, not fixed.

No training, RF-DETR inference, full-dataset optimisation, commit or push occurred.

## Documentation changes

Completed Google-style docstrings in `full_grid_search_apple.py` for
`load_module`, `file_hash`, `tree_partition`, `combinations`, `append_row`,
`load_rows`, `numeric`, `score`, `select_best`, `evaluate_configuration`, `main`,
and the nested `load_one` and `get_frames` helpers. Other source modules already
had docstrings for all discovered functions/classes; they were left unchanged.

Updated the root, source, output and documentation READMEs and three Sphinx guides.
Added the independent test suite, fixtures, report runner, test README and real report.
Regenerated API RST/HTML, including winter inference, and removed stale generated
pages for the already absent optimiser. The build configuration now avoids emitting
whitespace-only lines in generated API RST. `git diff --check` passed.

The comparison script's generated report still embeds historical `treeid` commands
and fixed historical prose. Those strings are executable statements and were left
unchanged; use the current README commands and independently verify report claims.
