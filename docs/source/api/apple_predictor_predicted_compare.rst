apple_predictor_predicted_compare
=================================

.. py:module:: apple_predictor_predicted_compare

Source: ``src/apple_predictor_predicted_compare.py``.

Statically extracted signatures and existing docstrings; no module execution.

Use improved U-Net branch predictions for traditional apple detection, then compare.

Run from the repository root with the mma3001 environment:
    python src/apple_predictor_predicted_compare.py

Default workflow: load a trusted improved U-Net checkpoint, infer a branch mask for
*each summer RGB-D frame*, put the checkpoint-sized square prediction (256 by
default) back into the central square
of the original RGB image, run the UNCHANGED traditional apple candidate detector,
export compatible prediction CSVs, and immediately compare them against the EXISTING
RF-DETR pseudo-reference CSVs. The RF-DETR model is NOT run or modified.

Optional --also-gt-control runs an equivalent traditional detection experiment with
GT branch masks using the SAME centre-square alignment. This is a more controlled
comparison of predicted vs GT branches than using historical full-resize results.

IMPORTANT: On summer training/validation images this is not an independent test of
branch-network generalisation. RF-DETR CSVs are pseudo-ground truth, not apple GT.

.. py:function:: centre_square_mask(mask_2d: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray

   Project branch foreground into the original image's central square.

   Uses nearest-neighbour resize to min(H, W), then zero padding. Unlike
   RF-DETR's helper, there is no same-size shortcut: every mask is projected.
   Nonsquare 2D inputs are stretched to a square, not explicitly rejected.

   :param mask_2d: numpy.ndarray [Hm, Wm]; values >0 denote foreground.
   :param image_shape: tuple[int, int]; positive original (height, width), pixels.

   :returns: uint8 [H, W] mask encoded 0/255.
   :rtype: numpy.ndarray

   :raises ValueError: If mask_2d is not 2D or image_shape cannot be unpacked.
   :raises cv2.error: If input or resize dimensions are invalid.

.. py:function:: load_improved_model(checkpoint_path: Path, requested_device: str)

   Load a local improved U-Net checkpoint without training or downloads.

   Reconstructs winter_inference.ImprovedUNet with weights=None, strictly loads
   state keys, moves to the selected device and enters evaluation mode.
   weights_only=True restricts deserialisation; use trusted local files.

   :param checkpoint_path: pathlib.Path to a checkpoint dictionary containing
                           model_state_dict and optionally input_size, or a bare state dict.
   :param requested_device: str; auto, cpu or cuda as constrained by the CLI.

   :returns: Model, torch module, winter_inference module, torch.device,
             integer input side (checkpoint input_size or 256), and checkpoint dict.
   :rtype: tuple

   :raises FileNotFoundError: If the checkpoint is absent.
   :raises TypeError: If the loaded checkpoint is not a dictionary.
   :raises RuntimeError: If CUDA is unavailable, state keys differ or loading fails.
   :raises ImportError: If torch/torchvision or winter_inference cannot be imported.
   :raises ValueError: If checkpoint input_size cannot be converted to int.

.. py:function:: make_branch_prediction(rgb_path, depth_path, model_info, threshold: float)

   Infer a square branch mask with training-compatible RGB-D scaling.

   Reopens both images through winter.preprocess: centre crop, bilinear RGB,
   nearest-neighbour L depth, ImageNet RGB normalisation and depth/255. Uses
   inference_mode; the supplied model must already be in evaluation mode.

   :param rgb_path: str or pathlib.Path to the RGB image.
   :param depth_path: str or pathlib.Path to its registered depth image.
   :param model_info: Six-element tuple returned by load_improved_model.
   :param threshold: float in [0, 1]; inclusive sigmoid probability cutoff.

   :returns: uint8 [S, S] 0/255 prediction; S from model_info.
   :rtype: numpy.ndarray

   :raises OSError: If either image cannot be decoded.
   :raises RuntimeError: If inference or device transfer fails.
   :raises ValueError: If tuple unpacking or preprocessing dimensions fail.

.. py:function:: read_predicted_mask(path: Path) -> np.ndarray

   Read a cached square grayscale mask and threshold intensities >127.

   An all-zero mask is valid. A mask whose maximum is exactly 1 is rejected;
   other grayscale encodings are accepted and thresholded, not validated as
   strictly binary 0/255. Provenance and threshold compatibility are not checked.

   :param path: pathlib.Path to an existing square PNG, normally encoded 0/255.

   :returns: uint8 [S, S] mask in {0, 255}.
   :rtype: numpy.ndarray

   :raises FileNotFoundError: If path is not a file.
   :raises ValueError: If dimensions are nonsquare or maximum intensity is 1.
   :raises OSError: If image decoding fails.

.. py:function:: detect_and_export(frames, source, args, model_info=None)

   Detect full-frame apple candidates using projected branch masks.

   Uses detector.find_apple_candidates unchanged. Overwrites CSVs, metadata
   and matching annotation PNGs; old unmatched PNGs can remain. Custom paths
   can overwrite other experiments, so keep output directories distinct.
   branch_ms includes inference/read, mask save and alignment; detection_ms
   excludes branch prediction. total_ms excludes final CSV/metadata export and
   model loading, and is not winter's model-only timing.

   :param frames: Sequence of (RGB Path, parsed info dict, depth Path, GT Path).
   :param source: str; predicted or gt. GT is thresholded >127 before projection.
   :param args: argparse.Namespace from main; supplies output/mask paths,
                checkpoint, season, reuse_masks and probability threshold.
   :param model_info: load_improved_model tuple; required for fresh predictions,
                      unused for GT and cached-mask reuse.

   :returns: Number of processed frames. Writes counts/instances/timing CSVs,
             run_metadata.json and flat annotation PNGs; fresh predictions also write
             <RGB-stem>_branch.png square masks.
   :rtype: int

   :raises ValueError: If source or RGB/depth dimensions are invalid.
   :raises FileNotFoundError: If a required GT or cached mask is absent.
   :raises OSError: If image/CSV/metadata reading or writing fails.
   :raises RuntimeError: If branch inference fails.

   .. note::

      Reused masks are not fingerprinted. Metadata's checkpoint and threshold
      describe the requested configuration, not verified cached-mask provenance.

.. py:method:: detect_and_export.write_csv(filename, columns, rows)

   Overwrite one experiment CSV, including its header for empty rows.

   :param filename: str basename relative to the enclosing output_dir.
   :param columns: Sequence[str] specifying exact CSV column order.
   :param rows: Iterable[dict] compatible with columns.

   :returns: None.

   :raises OSError: If file creation or writing fails.
   :raises ValueError: If a row has unexpected keys.

.. py:function:: run_comparison(traditional_dir: Path, output_dir: Path, args, source: str)

   Run saved-CSV comparison and adapt its report prose for this workflow.

   Matching and metric calculation belong to compare_apple_detectors.py.
   Report replacements depend on paragraph prefixes; they do not validate
   historical RF-DETR branch settings. Retain the reference run's provenance.

   :param traditional_dir: pathlib.Path containing counts and instances CSVs.
   :param output_dir: pathlib.Path for comparison reports, CSVs, plots and examples.
   :param args: argparse.Namespace with reference_dir, rgb_dir and max_distance.
   :param source: str; predicted or gt, used for report interpretation.

   :returns: First summary_metrics.csv row; prints selected metrics.
   :rtype: dict[str, str]

   :raises subprocess.CalledProcessError: If the comparison subprocess fails.
   :raises OSError: If report or summary cannot be accessed.
   :raises StopIteration: If the summary has no data row.

.. py:function:: build_frames(args)

   Discover season-filtered RGB-D pairs, optionally requiring paired GT.

   Searches lowercase PNG extensions nonrecursively, skips invalid names and missing
   depth with warnings. GT is required for gt source or also_gt_control, so
   enabling the control can reduce the predicted experiment's frame set.

   :param args: argparse.Namespace with rgb_dir, depth_dir, gt_dir, season,
                branch_source, also_gt_control and optional positive max_images.

   :returns: Sorted (RGB Path, parsed info dict, depth Path, GT Path),
             limited to the first max_images if supplied.
   :rtype: list[tuple]

   :raises RuntimeError: If no eligible complete pairs remain.

.. py:function:: main(argv=None)

   Parse options, produce predictions and optionally run comparisons.

   Defaults to summer improved-U-Net branches and fixed existing RF-DETR CSVs.
   --branch-source gt uses gt_output_dir/gt_comparison_dir, not predicted paths.
   --reuse-masks skips model loading; --skip-comparison needs no reference CSVs.
   A requested GT control uses the same discovered frame set and apple filters.

   :param argv: Sequence[str] or None; CLI arguments, or sys.argv[1:] when None.

   :returns: None. Writes experiment outputs and prints progress/summary metrics.

   :raises SystemExit: For help, invalid threshold/image limit, redundant control,
       invalid CLI options or missing required reference CSVs.
   :raises RuntimeError: If no paired frames exist or model execution fails.
   :raises OSError: If required inputs/outputs cannot be accessed.
   :raises subprocess.CalledProcessError: If saved-CSV comparison fails.

