Branch Segmentation
===================

Technical documentation for the RGB-D tree-branch segmentation experiments
in the Apple Tree Localiser project.

This documentation currently covers the three segmentation approaches:

* pixel-wise multilayer perceptron (MLP);
* basic ResNet-34 U-Net;
* improved ResNet-34 U-Net with ASPP, attention gates, and residual decoder blocks.

Apple detection and RF-DETR documentation is intentionally out of scope for
this first version.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   overview
   data_and_preprocessing
   models
   training_and_evaluation
   usage
