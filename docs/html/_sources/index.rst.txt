Apple Tree Localiser - MMA3001
==============================

Documentation for RGB-D branch segmentation and 2D apple localisation, including
MLP/U-Net models, traditional detection, RF-DETR, comparison, optimisation and
preprocessing. Standalone detectors default to GT; the predicted-branch driver
connects improved-U-Net masks to traditional detection.

The API reference uses existing docstrings and statically extracted signatures.
Building this site does not import any project source module.

.. toctree::
   :maxdepth: 2
   :caption: Project documentation

   reviewer_start
   overview
   workflows
   predicted_branches
   data_and_preprocessing
   models
   training_and_evaluation
   usage
   api/index

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
