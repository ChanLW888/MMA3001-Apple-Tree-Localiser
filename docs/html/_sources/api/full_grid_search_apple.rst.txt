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
A resume-safe all_experiments.csv is appended after each successful configuration.
By default tune on development trees, then compare the chosen model with the
baseline on separate held-out trees. No annotated PNGs or per-run CSVs are made.

.. py:function:: load_module(path, alias)

   No docstring is present in the source.

.. py:function:: file_hash(path)

   No docstring is present in the source.

.. py:function:: tree_partition(filename)

   No docstring is present in the source.

.. py:function:: combinations()

   No docstring is present in the source.

.. py:function:: append_row(path, row)

   No docstring is present in the source.

.. py:function:: load_rows(path)

   No docstring is present in the source.

.. py:function:: numeric(row, key, maximize=True)

   No docstring is present in the source.

.. py:function:: score(row, metric)

   No docstring is present in the source.

.. py:function:: select_best(rows, criterion)

   No docstring is present in the source.

.. py:function:: evaluate_configuration(params, frames, names, predictor, comparator, reference, threshold)

   No docstring is present in the source.

.. py:function:: main()

   No docstring is present in the source.

.. py:method:: main.load_one(path, info)

   No docstring is present in the source.

.. py:method:: main.get_frames(target_split)

   No docstring is present in the source.

