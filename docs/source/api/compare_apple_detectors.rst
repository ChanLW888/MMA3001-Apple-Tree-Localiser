compare_apple_detectors
=======================

.. py:module:: compare_apple_detectors

Source: ``src/compare_apple_detectors.py``.

Statically extracted signatures and existing docstrings; no module execution.

Compare traditional predictions with RF-DETR pseudo-ground truth.

Matches exact basenames and original-image centroids from saved CSVs. One-to-one
matching maximises valid pairs before minimising distance. Agreement is not
independently verified accuracy. Runs no detector inference. Generated report prose
includes fixed historical statements that this program does not verify.

.. rubric:: Example

python src/compare_apple_detectors.py --max-distance 30 --thresholds 10 20 30 40 50


.. py:function:: read_csv(path)

   Read headers and rows from a possibly BOM-prefixed CSV.
   
   :param path: str or pathlib.Path to UTF-8 CSV.
   
   :returns: Header names and unconverted rows.
   :rtype: tuple[list[str], list[dict[str, str]]]
   
   :raises ValueError: If no header exists.
   :raises OSError: If input reading fails.

.. py:function:: load_instances(path)

   Validate and group inclusive-coordinate instance records.
   
   IDs are positive and unique per image. Finite nonnegative coordinates must contain
   the centroid; image bounds are checked later by main. Confidence may be blank or
   NaN; finite scores must be in [0, 1]. Duplicate geometry is retained.
   
   :param path: str or pathlib.Path; CSV containing REQUIRED columns.
   
   :returns:
   
             Grouped records and retained
                 duplicate-geometry warnings.
   :rtype: tuple[dict[str, list[dict]], list[dict]]
   
   :raises ValueError: If headers, IDs, geometry, area, overlap or confidence are
       invalid.
   :raises OSError: If CSV reading fails.

.. py:function:: load_counts(path, instances)

   Validate processed-image counts against instance rows.
   
   Traditional identity comes from Annotated_Image; RF-DETR uses image. Counts
   distinguish zero detections from unprocessed images.
   
   :param path: str or pathlib.Path; traditional or RF-DETR counts CSV.
   :param instances: Mapping[str, list[dict]] grouped by exact image basename.
   
   :returns:
   
             Nonnegative counts, including processed zero-detection
                 frames.
   :rtype: dict[str, int]
   
   :raises ValueError: If headers, integer counts, unique names or instance/count
       agreement fail.
   :raises OSError: If CSV reading fails.

.. py:function:: match_centroids(reference, traditional, threshold)

   Maximise valid one-to-one pairs, then minimise summed distance.
   
   Hungarian assignment adds unmatched dummy columns. A penalty larger than any valid
   total prioritises cardinality over distance. RF-DETR is pseudo-ground truth, not
   independently verified apple annotation.
   
   :param reference: Sequence[dict] with RF-DETR cx_px and cy_px.
   :param traditional: Sequence[dict] with traditional cx_px and cy_px.
   :param threshold: float; finite nonnegative inclusive Euclidean tolerance in
                     original-image pixels.
   
   :returns:
   
             Reference index, traditional index and
                 distance; empty if either sequence is empty.
   :rtype: list[tuple[int, int, float]]
   
   :raises ValueError: If threshold is negative/nonfinite or assignment costs are
       invalid.

.. py:function:: bbox_iou(a, b)

   Measure overlap of two inclusive xyxy pixel boxes.
   
   The +1 terms count both boundary pixels. IoU is descriptive, not a centroid matching
   criterion.
   
   :param a: dict with numeric x1, y1, x2, y2; valid nonempty box.
   :param b: dict with the same inclusive-coordinate fields.
   
   :returns: Intersection/union in [0, 1] for valid boxes.
   :rtype: float
   
   :raises KeyError: If a coordinate is missing.

.. py:function:: agreement(tp, fp, fn)

   Compute micro agreement relative to RF-DETR.
   
   Error labels express disagreement with model predictions, not verified real-world
   errors.
   
   :param tp: int; nonnegative accepted match count.
   :param fp: int; nonnegative unmatched traditional detections.
   :param fn: int; nonnegative unmatched reference detections.
   
   :returns:
   
             Reference-relative counts, precision, recall and F1;
                 undefined ratios return zero.
   :rtype: dict[str, int | float]

.. py:function:: evaluate(images, ref, pred, threshold)

   Summarise fixed detections over common processed images.
   
   Missing instance keys imply zero detections. Count error is traditional minus
   reference. Distance/IoU statistics condition on accepted matches; absent samples
   yield None. Global agreement is micro; image_macro_f1 averages per-image F1.
   
   :param images: Nonempty reusable collection[str] of common exact basenames,
                  including zero-count frames.
   :param ref: Mapping[str, list[dict]] of RF-DETR pseudo-reference records.
   :param pred: Mapping[str, list[dict]] of traditional records.
   :param threshold: float; finite nonnegative inclusive tolerance in pixels.
   
   :returns:
   
             Per-image metrics, matched/unmatched
                 pairs and aggregate summary.
   :rtype: tuple[list[dict], list[dict], dict]
   
   :raises ValueError: If matching tolerance is invalid.
   :raises ZeroDivisionError: If images is empty.

.. py:method:: evaluate.pair(i, j, d, status)

   Build a matched or unmatched record for the current image.
   
   :param i: int or None; reference index, absent for unmatched prediction.
   :param j: int or None; prediction index, absent for unmatched reference.
   :param d: float or str; accepted distance in pixels, or empty string.
   :param status: str; matched, unmatched_reference or unmatched_traditional.
   
   :returns:
   
             Image, IDs, distance, conditional inclusive-box IoU and
                 status.
   :rtype: dict
   
   :raises IndexError: If a supplied index is outside the current records.

.. py:function:: write_csv(path, rows, fields=None)

   Write dictionaries using explicit or inferred columns.
   
   :param path: str or pathlib.Path; parent must exist.
   :param rows: list[dict]; records compatible with fields.
   :param fields: Sequence[str] or None; explicit order, required for empty rows.
   
   :returns: None.
   
   :raises IndexError: If rows is empty and fields is absent/empty.
   :raises ValueError: If rows have unexpected columns.
   :raises OSError: If writing fails.

.. py:function:: main()

   Compare saved CSVs without rerunning either detector.
   
   Writes metrics, sensitivity, coverage, examples, SHA-256 manifest and annotation
   inventory. Defaults to apple_rfdetr_instances and apple_detection under outputs.
   Current source is not proof of historical CSV provenance. Embedded report prose is
   fixed program data, not verified experiment history.
   
   :returns: None.
   
   :raises SystemExit: If CLI arguments are invalid or --help is requested.
   :raises ValueError: If CSVs, counts, geometry, coverage or tolerance are invalid.
   :raises OSError: If inputs or report/plot outputs cannot be accessed.

