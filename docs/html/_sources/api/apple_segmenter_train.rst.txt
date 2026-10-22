apple_segmenter_train
=====================

.. py:module:: apple_segmenter_train

Source: ``src/apple_segmenter_train.py``.

Statically extracted signatures and existing docstrings; no module execution.

Train the RF-DETR medium apple instance-segmentation model.

Uses cleaned COCO train/valid splits in data/apple_tree_clean. Requests 50 epochs and
batch size 1; remaining defaults belong to rfdetr. Writes outputs/rfdetr_apple_tree.
Import checks paths and prints configuration; training requires main. Cached pretrained
weights may be needed offline.

:raises FileNotFoundError: If data folders or COCO files are missing.
:raises ImportError: If rfdetr is unavailable.

.. rubric:: Example

python src/apple_segmenter_train.py


.. py:function:: main()

   Train RFDETRSegMedium on cleaned apple COCO splits.
   
   Uses 50 epochs, batch size 1 and library defaults otherwise. Writes
   outputs/rfdetr_apple_tree. Module checks dataset paths at import. Construction may
   retrieve pretrained weights if uncached.
   
   :returns: None.
   
   :raises OSError: If data, pretrained weights or output cannot be accessed.
   :raises RuntimeError: If training fails.

