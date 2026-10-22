apple_segmenter_v1
==================

.. py:module:: apple_segmenter_v1

Source: ``src/apple_segmenter_v1.py``.

Statically extracted signatures and existing docstrings; no module execution.

Run branch-guided RF-DETR apple instance segmentation.

Reads outputs/rfdetr_apple_tree/checkpoint_best_total.pth and writes
outputs/apple_rfdetr_instances. Defaults to summer branch GT (>127) with central
placement. Predicted mode only reads exported <RGB-stem>_branch.png masks. Does not run
branch inference or consume depth. Model detections are not verified apple annotations.

.. rubric:: Example

python src/apple_segmenter_v1.py


.. py:function:: parse_rgb_filename(path: Path)

   Parse tree, season and time from an RGB PNG basename.
   
   :param path: pathlib.Path; expected
                R<digits>N<digits>_<Summer|Winter>_RGB-<time>.png.
   
   :returns:
   
             tree, season and time in original case, or None for
                 unmatched names.
   :rtype: dict[str, str] or None

.. py:function:: corresponding_branch_path(image_path: Path, info: dict) -> Path

   Resolve a GT or externally exported predicted branch mask.
   
   Default source is GT. This does not run a branch network or verify predicted-mask
   provenance.
   
   :param image_path: pathlib.Path to the RGB frame.
   :param info: dict[str, str]; tree, season and time.
   
   :returns: GT path or <RGB-stem>_branch.png in PREDICTED_BRANCH_DIR.
   :rtype: pathlib.Path
   
   :raises ValueError: If BRANCH_SOURCE is not gt or predicted.
   :raises KeyError: If GT metadata is incomplete.

.. py:function:: align_branch_mask(mask: np.ndarray, image_shape, alignment: str)

   Place a binary branch mask in original RGB coordinates.
   
   Thresholds >127 before resizing. Same-size masks return directly. center_crop
   expands to min(H, W) square and centres on a zero canvas; full_resize stretches over
   the frame. Neither verifies true GT geometry. Input encoded as 0/1 becomes empty.
   
   :param mask: numpy.ndarray; grayscale [Hm, Wm] or BGR mask, expected 0/255
                encoding.
   :param image_shape: tuple[int, ...]; starts with original (height, width).
   :param alignment: str; full_resize or center_crop when shapes differ.
   
   :returns: uint8 [H, W] branch mask in {0, 1}.
   :rtype: numpy.ndarray
   
   :raises ValueError: If shapes differ and alignment is unsupported.
   :raises cv2.error: If conversion or resizing fails.

.. py:function:: make_branch_zone(branch_mask: np.ndarray)

   Optionally retain the largest branch component, then dilate.
   
   BRANCH_DILATION is ellipse diameter in source-image pixels. Largest-component
   filtering is disabled by default.
   
   :param branch_mask: numpy.ndarray; aligned [H, W], foreground >0.
   
   :returns: uint8 [H, W] proximity zone in {0, 1}.
   :rtype: numpy.ndarray
   
   :raises ValueError: If BRANCH_DILATION <1.
   :raises cv2.error: If component analysis or morphology fails.

.. py:function:: filter_instances(detections, image_shape, zone)

   Retain whole RF-DETR masks with sufficient branch-zone overlap.
   
   Missing masks/confidences yield no instances. Current constants require confidence
   >=0.20, mask >0.50, area >=1 and overlap >=0.05. Uses nearest-neighbour resizing. No
   class-ID filtering occurs; checkpoint is assumed apple-only. Masks remain whole, not
   clipped to the zone.
   
   :param detections: RF-DETR/supervision detections with mask [N, Hm, Wm] and
                      confidence [N], or None.
   :param image_shape: tuple[int, ...]; original (H, W, ...) shape.
   :param zone: numpy.ndarray; aligned binary [H, W] proximity region.
   
   :returns:
   
             mask (bool [H, W]), confidence, area (pixels), overlap,
                 inclusive xyxy bbox and mean (x, y) centroid.
   :rtype: list[dict]
   
   :raises ValueError: If a mask is not 2D or zone cannot broadcast.
   :raises cv2.error: If resizing fails.

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

.. py:function:: save_outputs(image, image_path, branch, zone, instances)

   Write cyan annotations, uint16 instance IDs and branch-zone PNGs.
   
   ID 0 is background; instances start at 1. Later masks overwrite earlier IDs at
   overlaps; CSV areas still describe whole instances.
   
   :param image: numpy.ndarray; uint8 [H, W, 3] BGR.
   :param image_path: pathlib.Path; stem names saved files.
   :param branch: numpy.ndarray; accepted but unused.
   :param zone: numpy.ndarray; [H, W] binary 0/1 proximity mask.
   :param instances: list[dict] from filter_instances.
   
   :returns:
   
             Annotated image,
                 instance-ID mask and zone image.
   :rtype: tuple[pathlib.Path, pathlib.Path, pathlib.Path]
   
   :raises ValueError: If count exceeds uint16 capacity.
   :raises OSError: If directories or image files cannot be written.
   :raises cv2.error: If encoding or drawing fails.

.. py:function:: main()

   Load RF-DETR and export branch-guided apple predictions.
   
   Defaults to summer branch GT with central placement. Skips missing/unreadable inputs
   and unexpected names. Recreates CSVs even for START_FRAME >1. model.predict reads
   RGB paths; no depth input is used. Initialisation may require cached auxiliary
   weights.
   
   :returns: None.
   
   :raises FileNotFoundError: If RGB directory or checkpoint is missing.
   :raises ImportError: If rfdetr cannot be imported.
   :raises ValueError: If overlap/start-frame settings, mask rank or alignment are
       invalid.
   :raises RuntimeError: If no selected PNGs exist or model execution fails.
   :raises OSError: If outputs cannot be written.

.. py:method:: main.sync_cuda()

   Synchronise pending CUDA work for inference timing.
   
   Does nothing without CUDA; barriers keep asynchronous kernels inside the timed
   interval.
   
   :returns: None.
   
   :raises RuntimeError: If CUDA synchronisation fails.

