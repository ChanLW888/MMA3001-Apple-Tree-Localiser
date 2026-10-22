Source API reference
====================

All maintained Python modules under ``src/`` are parsed using Python's AST parser.
Sphinx Napoleon formats existing Google-style docstrings. Signatures, classes,
methods, nested helpers, parameters, returns and exceptions are included where
documented. Missing prose is labelled rather than invented. Imported third-party
members, inherited APIs and dynamically created objects are not expanded; runtime
behaviour and annotation resolution are not verified. Constants are not evaluated.

The organiser copies files on import; COCO preparation and RF-DETR training check
paths on import. Static extraction avoids these side effects for every module.

.. toctree::
   :maxdepth: 1

   apple_predictor_predicted_compare
   apple_predictor_v1
   apple_segmenter_train
   apple_segmenter_v1
   compare_apple_detectors
   dataset_organiser
   full_grid_search_apple
   preprocess_roboflow_dataset
   train_baseline
   train_improved_unet
   train_unet
   winter_inference
