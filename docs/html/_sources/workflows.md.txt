# Apple workflows and environments

The project combines RGB-D branch segmentation with separate 2D apple localisation experiments. Standalone detectors default to branch GT. The [predicted branch driver](predicted_branches.md) runs the improved U-Net before traditional detection, with an optional aligned-GT control. RF-DETR predictions used for comparison are pseudo-ground truth, not verified apple annotations.

## Two pip environments

Run commands from the repository root. Activate with `conda activate mma3001` or `conda activate rfdetr`. Use the root README installation steps to select CPU or CUDA wheels before installing the requirements. Python 3.13 is used for main workflows and Python 3.11 for RF-DETR.

| Environment | Requirements | Workflows |
|---|---|---|
| `mma3001` | `requirements.txt` | Branch training, traditional detection, comparison, optimisation, dataset utilities; documentation with existing Sphinx/MyST |
| `rfdetr` | `requirements-rfdetr.txt` | RF-DETR segmentation training and inference |

The two detector stacks need not coexist in one environment. Datasets and checkpoints are supplied separately. Clean Windows installation and hardware verification evidence is recorded separately; other operating systems are untested.

## API navigation

- Branch models/training: [MLP](api/train_baseline.rst), [U-Net](api/train_unet.rst), [extended U-Net](api/train_improved_unet.rst).
- Traditional detection: [apple predictor](api/apple_predictor_v1.rst), [predicted driver](api/apple_predictor_predicted_compare.rst).
- RF-DETR: [training](api/apple_segmenter_train.rst), [inference](api/apple_segmenter_v1.rst).
- Saved-prediction [comparison](api/compare_apple_detectors.rst).
- Parameter experiments: [full grid search](api/full_grid_search_apple.rst).
- Dataset utilities: [orchard organiser](api/dataset_organiser.rst), [COCO preprocessing](api/preprocess_roboflow_dataset.rst).

Comparison reads saved predictions; parameter experiments execute detection when deliberately run. Documentation building only parses source text. No training, inference or experiment is needed to build this site.

## Licence boundary

The root MIT licence applies only to original project code owned by Chan Lik Wai. It grants no ownership or redistribution rights for university orchard datasets, Roboflow datasets, third-party weights, RF-DETR code or pretrained weights, or other externally licensed materials. Their own terms and permissions continue to apply. Dataset provenance and redistribution permission remain to be confirmed separately.
