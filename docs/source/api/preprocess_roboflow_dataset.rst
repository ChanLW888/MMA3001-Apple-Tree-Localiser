preprocess_roboflow_dataset
===========================

.. py:module:: preprocess_roboflow_dataset

Source: ``src/preprocess_roboflow_dataset.py``.

Statically extracted signatures and existing docstrings; no module execution.

Convert COCO apple-* categories into one apple class.

Maps data/apple tree.v1i.coco-segmentation/train and export to
data/apple_tree_clean/train and valid. Retains all image records, including missing
images whose copies are skipped. Annotation origin and independence from orchard
evaluation require verification. Source path checks run on import.

:raises FileNotFoundError: If configured folders or annotation files are absent.

.. rubric:: Example

python src/preprocess_roboflow_dataset.py


.. py:function:: process_split(source_dir, annotation_file, output_dir)

   Copy a COCO split and merge apple-* categories into class 1.

   Keeps all images, including negative examples, and preserves segmentation geometry.
   Missing image records remain in COCO despite skipped copies; check output
   completeness separately. Reassigns annotation IDs sequentially.

   :param source_dir: pathlib.Path containing images.
   :param annotation_file: str or pathlib.Path; COCO JSON with images, annotations and
                           categories.
   :param output_dir: pathlib.Path; split destination created if needed.

   :returns: images, apple_instances, copied_images and missing_images.
   :rtype: dict[str, int]

   :raises RuntimeError: If no category starts with apple- (case-insensitive).
   :raises json.JSONDecodeError: If annotation JSON is invalid.
   :raises KeyError: If required category/image fields are absent.
   :raises OSError: If reading, copying or writing fails.

.. py:function:: main()

   Prepare train and valid apple-only COCO splits and print counts.

   Maps source train to train and export to valid. Module-level checks require source
   paths even on import.

   :returns: None.

   :raises RuntimeError: If no apple-* categories exist.
   :raises OSError: If annotation reading, copying or writing fails.
   :raises json.JSONDecodeError: If JSON is malformed.

