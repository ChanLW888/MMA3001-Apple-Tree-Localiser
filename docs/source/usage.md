# Usage

Run commands from the **repository root**. The examples below use common
filenames from the branch-segmentation workflow; change the filename if your
current `src/` folder uses a different name.

## Check available arguments

```bash
python src/train_pixel_mlp_crop_trainval.py --help
python src/train_resnet34_unet_crop_trainval.py --help
python src/train_improved_unet_crop_trainval.py --help
```

## Train a model

Supply the path to the organised dataset using `--data-root`. For example:

```bash
python src/train_resnet34_unet_crop_trainval.py --data-root "C:/path/to/Organised"
```

Replace the placeholder with the actual path on your machine. Review the
script's `--help` output before training to see its supported options and
default output directory.

## Build this documentation

Install the documentation dependencies and generate HTML:

```bash
python -m pip install -r docs/requirements-docs.txt
python -m sphinx -b html docs/source docs/_build/html
```

Open `docs/_build/html/index.html` in a web browser.

This branch-only documentation build does not import or execute the project's
apple detection or RF-DETR scripts.
