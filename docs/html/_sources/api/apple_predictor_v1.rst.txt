apple_predictor_v1
==================

.. py:module:: apple_predictor_v1

Source: ``src/apple_predictor_v1.py``.

Statically extracted signatures and existing docstrings; no module execution.

Detect red apples with depth validity and branch ground truth.

A 2D image-space baseline, not metric 3D localisation. Branch GT (>0) is stretched to
the full RGB grid and dilated. RGB/depth must be aligned. Writes annotation PNGs,
counts, instances, timings and metadata under outputs/apple_detection.

.. rubric:: Example

python src/apple_predictor_v1.py


.. py:function:: parse_rgb_filename(path)

   Parse tree, season and time from an RGB PNG basename.

   :param path: pathlib.Path; expected
                R<digits>N<digits>_<Summer|Winter>_RGB-<time>.png.

   :returns:

             tree, season and time in original case, or None for
                 unmatched names.
   :rtype: dict[str, str] or None

.. py:function:: load_rgb(path)

   Read an image in RGB channel order.

   :param path: str or pathlib.Path; readable image path.

   :returns: uint8 [H, W, 3] RGB in [0, 255].
   :rtype: numpy.ndarray

   :raises OSError: If image opening or decoding fails.

.. py:function:: load_depth(path)

   Read stored depth intensities without metric conversion.

   :param path: str or pathlib.Path; readable depth image.

   :returns:

             [H, W] in stored dtype; first channel is used for
                 multichannel images.
   :rtype: numpy.ndarray

   :raises OSError: If image opening or decoding fails.

.. py:function:: load_gt(path, target_shape)

   Load branch ground truth and stretch it to the RGB grid.

   Nearest-neighbour interpolation preserves labels. Full-frame stretching differs from
   RF-DETR default central placement; it does not verify annotation geometry.

   :param path: str or pathlib.Path; branch GT image, not apple-instance annotation.
   :param target_shape: tuple[int, int]; positive original (height, width) in pixels.

   :returns:

             uint8 [H, W] grayscale GT; foreground is interpreted
                 downstream as >0.
   :rtype: numpy.ndarray

   :raises OSError: If GT cannot be decoded.
   :raises cv2.error: If resize dimensions are invalid.

.. py:function:: detect_red_mask(rgb, depth)

   Select red-dominant pixels with valid depth.

   Floating channel arithmetic avoids uint8 overflow. Thresholds are module constants;
   depth supplies validity, not 3D coordinates.

   :param rgb: numpy.ndarray; [H, W, 3] RGB intensities in [0, 255].
   :param depth: numpy.ndarray; aligned [H, W] values, valid when >0.

   :returns: uint8 [H, W] binary mask.
   :rtype: numpy.ndarray

   :raises ValueError: If depth cannot broadcast to the RGB grid.

.. py:function:: clean_red_mask(mask)

   Suppress isolated red noise and close small gaps.

   The ellipse is MORPH_KERNEL_SIZE pixels wide/high. Morphology may merge nearby
   regions; components are candidates rather than guaranteed apples.
   Closing can fill non-red or invalid-depth pixels after the initial validity
   test; final component pixels are not guaranteed individually depth-valid.

   :param mask: numpy.ndarray; uint8 [H, W] binary red mask.

   :returns: uint8 [H, W] opened then closed mask.
   :rtype: numpy.ndarray

   :raises cv2.error: If mask or kernel is invalid.

.. py:function:: create_branch_zone(gt)

   Dilate labelled branch pixels into a proximity region.

   BRANCH_DILATION is ellipse diameter in source-image pixels, not radius.
   The default entry point uses ground truth; the predicted comparison driver
   supplies network masks instead.

   :param gt: numpy.ndarray; aligned [H, W] branch mask, foreground >0; GT or predicted.

   :returns: uint8 [H, W] zone in {0, 1}.
   :rtype: numpy.ndarray

   :raises cv2.error: If morphology input or kernel is invalid.

.. py:function:: find_apple_candidates(rgb, depth, gt)

   Filter connected red components by geometry and branch proximity.

   Original-image coordinates: top-left origin, x right, y down. Overlap is
   intersection/whole-component area. Red score is mean R/(G+B+1), not calibrated
   confidence. No apple labels are read.

   :param rgb: numpy.ndarray; uint8 [H, W, 3] RGB.
   :param depth: numpy.ndarray; aligned [H, W] depth-validity intensities.
   :param gt: numpy.ndarray; aligned [H, W] branch mask; the gt argument name is historical.

   :returns:

             label, area (pixels), bbox (x, y, width, height), centroid (x,
                 y), circularity, aspect_ratio, branch_overlap and red_score.
   :rtype: list[dict]

   :raises ValueError: If RGB/depth/GT spatial shapes differ.
   :raises cv2.error: If image processing fails.

.. py:function:: render_apple_annotations(image_bgr, boxes, centroids, color_bgr)

   Render boxes, centres, indices and total count on a copy.

   zip truncates unequal box/centroid sequences; callers should supply one centre per
   box.

   :param image_bgr: numpy.ndarray; uint8 [H, W, 3] OpenCV BGR.
   :param boxes: Sequence of inclusive (x1, y1, x2, y2) boxes in pixels.
   :param centroids: Sequence of floating (x, y) centres paired with boxes.
   :param color_bgr: tuple[int, int, int]; BGR colour in [0, 255].

   :returns: Annotated BGR copy with original shape/dtype.
   :rtype: numpy.ndarray

   :raises cv2.error: If drawing arguments are invalid.

.. py:function:: draw_annotations(rgb, candidates)

   Convert candidate boxes to inclusive corners and draw orange labels.

   :param rgb: numpy.ndarray; uint8 [H, W, 3] RGB.
   :param candidates: list[dict] from find_apple_candidates.

   :returns: Annotated RGB copy for Pillow saving.
   :rtype: numpy.ndarray

   :raises KeyError: If bbox or centroid is missing.
   :raises cv2.error: If colour conversion or drawing fails.

.. py:function:: get_matching_files(rgb_path, info)

   Build depth and branch-GT paths for a parsed frame.

   :param rgb_path: pathlib.Path; accepted but unused.
   :param info: dict[str, str]; tree, season and time from parse_rgb_filename.

   :returns:

             Depth and GT paths; existence is not
                 checked.
   :rtype: tuple[pathlib.Path, pathlib.Path]

   :raises KeyError: If required metadata is missing.

.. py:function:: process_frame(rgb_path, info)

   Load one frame, detect candidates and save annotations.

   :param rgb_path: pathlib.Path to the RGB image.
   :param info: dict[str, str]; parsed tree, season and time.

   :returns:

             Frame identity, candidates, dimensions and
                 read_ms/detection_ms/save_ms; None if depth or GT is missing. main adds
                 total_ms.
   :rtype: dict or None

   :raises ValueError: If RGB/depth shapes differ.
   :raises OSError: If decoding or annotation saving fails.
   :raises cv2.error: If image processing fails.

.. py:function:: find_all_frames()

   Discover complete configured-season RGB/depth/GT triplets.

   Warns and skips unexpected names or missing companions. An absent RGB directory
   produces an empty list.

   :returns: Frames sorted by tree and timestamp string.
   :rtype: list[tuple[pathlib.Path, dict]]

.. py:function:: instance_rows(result)

   Export components in the shared instance-CSV convention.

   confidence is intentionally blank; red_score is an uncalibrated colour statistic.

   :param result: dict with image, instances, image_width and image_height.

   :Yields: *dict* --

            One row per candidate with 1-based apple_id, inclusive xyxy box and
                mean-pixel centroid in original RGB coordinates.

   :raises KeyError: If required result/candidate fields are absent.

.. py:function:: save_csv(results)

   Write counts, instances, timing CSVs and run metadata.

   Overwrites configured CSVs. Durations are milliseconds. Metadata records traditional
   parameters and GT full-resize alignment.

   :param results: list[dict]; process_frame results with total_ms added by main.

   :returns: None.

   :raises KeyError: If any export field, including total_ms, is missing.
   :raises OSError: If output creation or writing fails.

.. py:function:: main()

   Process configured frames and export traditional detector outputs.

   Configuration uses module constants, not CLI flags. Prints and returns without
   exports when no complete frames exist.

   :returns: None.

   :raises ValueError: If RGB/depth geometry differs.
   :raises OSError: If input reading or output writing fails.
   :raises cv2.error: If processing fails.

