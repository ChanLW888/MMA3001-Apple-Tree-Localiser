optimise_apple_predictor
========================

.. py:module:: optimise_apple_predictor

Source: ``src/optimise_apple_predictor.py``.

Statically extracted signatures and existing docstrings; no module execution.

One-factor-at-a-time traditional apple-detector sensitivity experiment.

Place alongside apple_predictor_v1.py and compare_apple_detectors.py in src/.
Run from repository root: python src/optimise_apple_predictor.py

Changes only in-memory traditional detector constants; the source, RF-DETR predictions,
branch dilation and overlap stay unchanged. No annotated PNGs are generated.

.. py:function:: load_module(path: Path, name: str)

   No docstring is present in the source.

.. py:function:: save_rows(path, fieldnames, rows)

   No docstring is present in the source.

.. py:function:: tree_partition(filename, test_fraction=0.2)

   Deterministic tree-ID grouping so observations of one tree stay together.

.. py:function:: main()

   No docstring is present in the source.

