# Usage

Run these commands from the repository root in an already provisioned environment. The filenames below exist in this checkout. Training/inference commands are instructions for future intentional runs and were not executed during documentation work.

## Branch options

```powershell
python src/train_baseline.py --help
python src/train_unet.py --help
python src/train_improved_unet.py --help
```

## Train separately

```powershell
python src/train_baseline.py --data-root data/Organised
python src/train_unet.py --data-root data/Organised
python src/train_improved_unet.py --data-root data/Organised
```

`--data-root` is required. Defaults use summer samples with branch GT; masks must already match the centre-cropped input geometry and size. Use `--output-dir` to select a run location and retain its settings. U-Nets support `--no-pretrained` to avoid requesting ImageNet initialisation. RF-DETR setup is separate; see the root README.

## Winter predictions

With checkpoints from the default branch training locations:

```powershell
python test/winter_inference.py --data-root data/Organised --mlp-checkpoint outputs/pixel_mlp_crop_trainval/pixel_mlp_best.pth --unet-checkpoint outputs/resnet34_unet_crop_trainval_output/resnet34_unet_best.pth --improved-checkpoint outputs/improved_resnet34_unet_output/improved_resnet34_unet_best.pth
```

These are expected generated paths, not files distributed with Git. Override paths for historical/custom runs. Checkpoints must be trusted and architecture-compatible. Supply `--input-size` explicitly when it differs from 256; geometry is not inferred from checkpoint metadata. `--threshold` defaults to 0.5, `--warmup` to 5. Output masks and model-only timing do not measure winter accuracy without GT.

## Build and view documentation

From the repository root, with Sphinx/MyST already installed in `mma3001`:

```powershell
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
```

Open `docs/_build/html/index.html` or the submission copy `docs/html/index.html` in a browser. Preserve all sibling files and asset directories. Static extraction reads source text and never imports project modules. No package installation, training or inference is required for the documentation build.
