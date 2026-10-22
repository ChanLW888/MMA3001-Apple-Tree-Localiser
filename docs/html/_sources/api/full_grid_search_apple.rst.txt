full_grid_search_apple
======================

.. py:module:: full_grid_search_apple

Source: ``src/full_grid_search_apple.py``.

Statically extracted signatures and existing docstrings; no module execution.

Full Cartesian grid search for traditional apple-detector parameters.

Place in src/ beside apple_predictor_v1.py and compare_apple_detectors.py.
Run from the repository root: python src/full_grid_search_apple.py

All 5,184 combinations are evaluated against the *existing* RF-DETR instance CSV.
Only traditional colour/component/morphology constants vary. Branch-region
constants, GT alignment, matching threshold and RF-DETR files are unchanged.
A resumable all_experiments.csv is appended after each successful configuration.
See main for fingerprint coverage and the held-out objective-change limitation.
By default tune on development trees, then compare the chosen model with the
baseline on separate held-out trees. No annotated PNGs or per-run CSVs are made.

.. py:function:: load_module(path, alias)

   Load a Python file under a supplied import alias.

   Executes module-level statements and registers the module in sys.modules;
   callers must use trusted, import-safe files.

   :param path: pathlib.Path to a Python source file.
   :param alias: str module name used in sys.modules.

   :returns: Executed module.
   :rtype: ModuleType

   :raises FileNotFoundError: If path is not a file.
   :raises Exception: Exceptions raised by the imported module propagate.

.. py:function:: file_hash(path)

   Fingerprint a complete file for run provenance.

   :param path: pathlib.Path to a readable file; bytes are read into memory.

   :returns: Hexadecimal SHA-256 digest.
   :rtype: str

   :raises OSError: If reading fails.

.. py:function:: tree_partition(filename)

   Assign every frame of one tree to the same deterministic split.

   :param filename: str basename whose prefix before the first underscore is a
                    tree ID. No filename validation is performed.

   :returns: ``test`` when the SHA-256-derived fraction is below 0.2,
             otherwise ``development``. The observed split need not be exactly 20%.
   :rtype: str

.. py:function:: combinations()

   Enumerate the Cartesian product of the current module GRID.

   Grid insertion order determines run IDs. The current seven parameter lists
   produce 4 * 4 * 4 * 3 * 3 * 3 * 3 = 5,184 configurations. Branch dilation
   and overlap are absent from the grid and remain fixed.

   :Yields: *tuple[str, dict]* -- Zero-based grid_NNNN ID and detector constants.
            Each call produces new parameter dictionaries.

.. py:function:: append_row(path, row)

   Append one configuration and its metrics to the result CSV.

   :param path: pathlib.Path; parent directory must exist.
   :param row: dict with SUMMARY_FIELDS-compatible keys. Missing values are blank.
               A header is written only when the file is empty.

   :returns: None.

   :raises ValueError: If row contains unexpected columns.
   :raises OSError: If opening or writing fails.

.. py:function:: load_rows(path)

   Read resumable results and require the exact current column order.

   :param path: pathlib.Path to all_experiments.csv.

   :returns: Raw CSV rows, or an empty list for an absent file.
             Run IDs and row values are not otherwise validated here.
   :rtype: list[dict[str, str]]

   :raises ValueError: If the header differs from SUMMARY_FIELDS.
   :raises OSError: If an existing file cannot be read.

.. py:function:: numeric(row, key, maximize=True)

   Convert a metric into a score where larger is preferred.

   :param row: Mapping of metric names to numeric values or numeric strings.
   :param key: str metric column.
   :param maximize: bool; negate finite values when False.

   :returns: Oriented score; negative infinity for absent, invalid or
             nonfinite values, independent of optimisation direction.
   :rtype: float

.. py:function:: score(row, metric)

   Map a supported optimisation objective to a larger-is-better score.

   :param row: Mapping containing evaluation metric columns.
   :param metric: One of f1, precision, recall, mae, rmse, centroid, bbox_iou,
                  count_bias. Errors/distances are minimised, agreement maximised,
                  and signed bias minimised in absolute magnitude.

   :returns: Oriented score; most invalid metrics become negative infinity.
   :rtype: float

   :raises KeyError: If metric is unknown or count_bias lacks its input column.
   :raises ValueError: If count_bias cannot be converted to float.
   :raises TypeError: If count_bias has an incompatible value type.

   .. note:: Unlike other objectives, count_bias does not reject nonfinite values.

.. py:function:: select_best(rows, criterion)

   Select the best row with deterministic baseline tie preference.

   :param rows: Nonempty sequence of result dictionaries with run_id, BASELINE
                parameter columns and the requested metric.
   :param criterion: str objective accepted by score.

   :returns: Original winning row. Equal scores prefer exact baseline values,
             then lexicographically smallest run_id.
   :rtype: dict

   :raises IndexError: If rows is empty.
   :raises KeyError: If an objective or required column is missing or unknown.
   :raises ValueError: If a parameter cannot be converted to float.
   :raises TypeError: If a numeric value has an incompatible type.

.. py:function:: evaluate_configuration(params, frames, names, predictor, comparator, reference, threshold)

   Evaluate cached synthetic or real frames without exporting images.

   Mutates predictor constants in memory; this helper does not restore them.
   The main search restores baseline constants in a finally block.

   :param params: Mapping of detector constant names to parameter values.
   :param frames: Iterable of (filename, info, rgb, depth, branch) tuples. RGB
                  has shape [H, W, 3]; aligned depth and branch masks have shape [H, W].
                  The info element is unused here.
   :param names: Nonempty reusable collection of exact basenames to evaluate.
   :param predictor: Module-like object exposing detector and instance_rows APIs.
   :param comparator: Module-like object exposing evaluate.
   :param reference: Mapping of basenames to saved RF-DETR instance records.
   :param threshold: float inclusive centroid tolerance in original RGB pixels.

   :returns: Aggregate comparison metrics and elapsed seconds
             for detection/instance conversion, excluding comparison evaluation.
   :rtype: tuple[dict, float]

   :raises ValueError: If spatial shapes or comparison tolerance are invalid.
   :raises ZeroDivisionError: If names is empty.
   :raises KeyError: If candidate/export fields are missing.

.. py:function:: main()

   Run or resume traditional parameter tuning against saved RF-DETR results.

   Reads command-line options through argparse; use --help for paths,
   selection objectives, by-tree/all splits, caching and partial-run options.

   :returns: None. Appends all_experiments.csv and rewrites best_by_metric.csv and
             metric_rankings.csv. Writes grid_metadata.json and, only after a complete
             by-tree search, heldout_comparison.csv if that file does not already exist.

   :raises ValueError: If baseline/fixed branch settings differ, common inputs or
       partitions are empty, or resume metadata/CSV columns are incompatible.
   :raises FileNotFoundError: If configured source or reference files are missing.
   :raises OSError: If image/CSV reading or result writing fails.
   :raises SystemExit: For argparse help or invalid arguments.

   .. note::

      Branch dilation is fixed at 100 pixels and minimum overlap at 0.05.
      Complete traditional RGB/depth/GT triplets are intersected with reference
      count basenames, preserving zero-count images. GT is stretched to RGB
      geometry. Images are cached in RAM unless --no-cache reloads each run.
      The no-cache path still materialises the entire split as a list; it is
      not a bounded-memory stream. The preceding iteration's current_frames
      remains referenced while the next list is built, so peak memory can
      include two split-sized image lists. The CLI help's RAM-reduction intent
      has not been established by this implementation.
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

.. py:method:: main.load_one(path, info)

   Load one aligned RGB/depth/branch frame for evaluation.

   :param path: pathlib.Path to an RGB image.
   :param info: dict with tree, season and time used to locate companions.

   :returns: Basename, info, RGB [H, W, 3], depth [H, W] and resized
             branch GT [H, W]; depth retains stored units.
   :rtype: tuple

   :raises ValueError: If RGB and depth dimensions differ.
   :raises OSError: If image decoding fails.
   :raises KeyError: If companion metadata is missing.

.. py:method:: main.get_frames(target_split)

   Load all discovered frames belonging to a split.

   :param target_split: str; development, test or all, matching groups.

   :returns: Loaded frame tuples in discovery order; empty if none.
   :rtype: list[tuple]

   :raises ValueError: If a frame has inconsistent RGB/depth shapes.
   :raises OSError: If image reading fails.

