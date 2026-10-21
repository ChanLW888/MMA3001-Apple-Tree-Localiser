#!/usr/bin/env python3
"""Train a pixel-wise RGB-D MLP for binary branch segmentation.

Float32 input [B, 4, S, S] is ImageNet-standardised RGB plus grayscale depth/255. GT [B,
1, S, S] uses intensity >127 as branch by default and must already match the centred
field of view. The MLP has no spatial context. Uses positive-weight-12 BCE, Adam, seeded
sample splitting and best-validation-Dice selection. Validation is not independent
testing.

Example:
    python src/train_baseline.py --data-root data/Organised
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# Replace each modality token with a common token to match corresponding files.
MODALITY_PATTERNS = {
    "rgb": re.compile(r"_RGB-", re.IGNORECASE),
    "depth": re.compile(r"_D-", re.IGNORECASE),
    "gt": re.compile(r"_GT-", re.IGNORECASE),
}

# ImageNet statistics keep RGB preprocessing consistent with the ResNet models.
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The branch class is sparse, so positive pixels receive a larger loss weight.
POSITIVE_CLASS_WEIGHT = 12.0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch generators.

    CUDA generators are seeded when available. Seeding does not guarantee deterministic
    GPU kernels.

    Args:
        seed: int; nonnegative NumPy-compatible random seed.

    Returns:
        None.

    Raises:
        ValueError: If NumPy rejects the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sample_key(path: Path, modality: str) -> str:
    """Build a modality-independent filename identifier.

    Args:
        path: pathlib.Path to an image; stem retains tree, season and time.
        modality: str; one of "rgb", "depth", or "gt".

    Returns:
        str: Stem with the first modality token replaced by "_TIME-".

    Raises:
        KeyError: If modality is unsupported.
    """
    return MODALITY_PATTERNS[modality].sub("_TIME-", path.stem, count=1)


def list_modality(root: Path, modality: str) -> list[Path]:
    """Find summer PNGs for one modality recursively.

    Args:
        root: pathlib.Path to the dataset directory.
        modality: str; "rgb", "depth", or "gt".

    Returns:
        list[pathlib.Path]: Sorted matching paths with summer in the stem.

    Raises:
        KeyError: If modality is unsupported.
    """
    pattern = MODALITY_PATTERNS[modality]
    return sorted(
        path
        for path in root.rglob("*.png")
        if pattern.search(path.stem) and "summer" in path.stem.lower()
    )


def map_unique(paths: list[Path], modality: str) -> dict[str, Path]:
    """Index paths while rejecting ambiguous sample identifiers.

    Args:
        paths: list[pathlib.Path] for one modality.
        modality: str; "rgb", "depth", or "gt".

    Returns:
        dict[str, pathlib.Path]: Identifiers mapped to unique files.

    Raises:
        RuntimeError: If paths produce duplicate identifiers.
        KeyError: If modality is unsupported.
    """
    result: dict[str, Path] = {}
    for path in paths:
        key = get_sample_key(path, modality)
        if key in result:
            raise RuntimeError(
                f"Duplicate {modality} identifier {key}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def build_samples(root: Path) -> list[dict[str, Any]]:
    """Pair complete summer RGB, depth and branch-GT triplets.

    Incomplete triplets are omitted. Filename agreement does not verify spatial
    alignment.

    Args:
        root: pathlib.Path; existing organised dataset root.

    Returns:
        list[dict]: Sorted records with key, rgb, depth and gt entries; paths are
            pathlib.Path objects.

    Raises:
        FileNotFoundError: If root does not exist.
        RuntimeError: If identifiers repeat or no complete triplets exist.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    rgb_map = map_unique(list_modality(root, "rgb"), "rgb")
    depth_map = map_unique(list_modality(root, "depth"), "depth")
    gt_map = map_unique(list_modality(root, "gt"), "gt")

    samples = [
        {
            "key": key,
            "rgb": rgb_path,
            "depth": depth_map[key],
            "gt": gt_map[key],
        }
        for key, rgb_path in rgb_map.items()
        if key in depth_map and key in gt_map
    ]

    print("\n========== DATASET PAIRING ==========")
    print(f"RGB={len(rgb_map)}, depth={len(depth_map)}, GT={len(gt_map)}")
    print(f"Complete triplets={len(samples)}")

    if not samples:
        raise RuntimeError("No complete RGB/depth/GT triplets found.")

    return sorted(samples, key=lambda sample: sample["key"])


def centre_crop_square(image: Image.Image) -> Image.Image:
    """Crop to the largest centred square without resampling.

    Args:
        image: PIL.Image.Image in any mode.

    Returns:
        PIL.Image.Image: Square crop of side min(width, height).
    """
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


class BranchDataset(Dataset):
    """Load aligned RGB-D inputs and binary branch targets.

    RGB and depth are centre-cropped and resized. GT is neither cropped nor resized; it
    must already describe that field of view at input_size. Channel order is R, G, B,
    then grayscale depth. Depth is image intensity, not calibrated metres.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        input_size: int = 256,
        mask_threshold: int = 127,
    ) -> None:
        """Store paired samples and preprocessing settings.

        Args:
            samples: list[dict] from build_samples; rgb, depth, gt and key entries.
            input_size: int; positive square side in pixels, default 256.
            mask_threshold: int; grayscale cutoff, normally [0, 255]; GT > cutoff is
                branch.

        Returns:
            None.
        """
        self.samples = samples
        self.input_size = input_size
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        """Return the number of complete samples.

        Returns:
            int: Number of stored records.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Read and preprocess one RGB-D and branch-GT sample.

        RGB is bilinearly resized, /255 then ImageNet standardised. Depth is converted
        to L, nearest-neighbour resized and /255. Targets use GT > mask_threshold; 1
        denotes branch and 0 background.

        Args:
            index: int; Python list index into samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: Float32 input [4, S, S], binary
                float32 target [1, S, S], and identifier; S is input_size.

        Raises:
            IndexError: If index is outside samples.
            KeyError: If a sample lacks a required entry.
            OSError: If an input cannot be opened or decoded.
            RuntimeError: If GT size differs from input_size.
        """
        sample = self.samples[index]
        size = (self.input_size, self.input_size)

        # Use bilinear interpolation for colour imagery to preserve smooth detail.
        with Image.open(sample["rgb"]) as image:
            rgb_image = centre_crop_square(image.convert("RGB"))
        rgb_image = rgb_image.resize(size, Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
        rgb = (rgb - RGB_MEAN) / RGB_STD

        # Nearest-neighbour resizing avoids inventing intermediate depth values.
        with Image.open(sample["depth"]) as image:
            depth_image = centre_crop_square(image.convert("L"))
        depth_image = depth_image.resize(size, Image.Resampling.NEAREST)
        depth = np.asarray(depth_image, dtype=np.float32) / 255.0

        # Model channel order is R, G, B, then grayscale depth.
        inputs = np.concatenate([rgb.transpose(2, 0, 1), depth[None]], axis=0)

        with Image.open(sample["gt"]) as image:
            gt_image = image.convert("L")
        if gt_image.size != size:
            raise RuntimeError(
                f"GT size {gt_image.size} != expected {size} for {sample['key']}. "
                "The GT must match input_size or preprocessing must be changed explicitly."
            )

        gt = np.asarray(gt_image, dtype=np.uint8)
        target = (gt > self.mask_threshold).astype(np.float32)[None]

        return (
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(target),
            sample["key"],
        )


class PixelMLP(nn.Module):
    """Classify each RGB-D pixel independently using dense layers.

    Flattening pools pixels into a batch without mixing neighbours; output is a raw
    branch logit per pixel.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
    ) -> None:
        """Construct hidden dense layers and scalar output.

        Args:
            in_features: int; positive features per pixel, normally 4.
            hidden_dims: tuple[int, ...]; positive hidden widths, default (64, 32,
                16).

        Returns:
            None.
        """
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features

        for hidden_size in hidden_dims:
            layers.extend(
                [nn.Linear(previous, hidden_size), nn.ReLU(inplace=True)]
            )
            previous = hidden_size

        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map independent pixel features to branch logits.

        Args:
            x: torch.Tensor; [B, in_features, H, W].

        Returns:
            torch.Tensor: Raw logits [B, 1, H, W]; sigmoid gives branch
                probabilities.

        Raises:
            RuntimeError: If feature count or dtype/device is incompatible.
        """
        batch_size, channels, height, width = x.shape
        # Move channels last so each dense-layer row contains one pixel, not a scanline.
        pixels = x.permute(0, 2, 3, 1).reshape(-1, channels)
        logits = self.network(pixels)
        return logits.reshape(batch_size, height, width).unsqueeze(1)


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy directly from branch logits.

    Positive-class weight 12 addresses sparse branch foreground.

    Args:
        logits: torch.Tensor; floating [B, 1, H, W] raw scores.
        targets: torch.Tensor; same shape/device, floating labels in {0, 1}.

    Returns:
        torch.Tensor: Scalar mean loss.

    Raises:
        ValueError: If target and logit sizes differ.
        RuntimeError: If tensor types or devices are incompatible.
    """
    positive_weight = torch.tensor(
        [POSITIVE_CLASS_WEIGHT], device=logits.device, dtype=logits.dtype
    )
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Aggregate validation loss and foreground pixel metrics.

    Sets evaluation mode and disables gradients. Uses sigmoid(logits) >= threshold and
    targets >=0.5. Epsilon 1e-8 stabilises ratios; empty foreground scores zero.

    Args:
        model: torch.nn.Module mapping [B, 4, H, W] to [B, 1, H, W] logits.
        loader: DataLoader yielding input, target and identifier batches.
        device: torch.device; model must already be on it.
        threshold: float; inclusive sigmoid cutoff, normally [0, 1].

    Returns:
        dict[str, float]: Sample-weighted loss and pixel-micro dice, iou, precision
            and recall.

    Raises:
        OSError: If dataset inputs cannot be read.
        RuntimeError: If dataset geometry or model execution fails.
    """
    model.eval()
    total_loss = 0.0
    true_positive = false_positive = false_negative = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += bce_loss(logits, targets).item() * inputs.size(0)

        predictions = torch.sigmoid(logits) >= threshold
        target_mask = targets >= 0.5
        true_positive += (predictions & target_mask).sum().item()
        false_positive += (predictions & ~target_mask).sum().item()
        false_negative += (~predictions & target_mask).sum().item()

    epsilon = 1e-8
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "dice": 2 * true_positive
        / (2 * true_positive + false_positive + false_negative + epsilon),
        "iou": true_positive
        / (true_positive + false_positive + false_negative + epsilon),
        "precision": true_positive / (true_positive + false_positive + epsilon),
        "recall": true_positive / (true_positive + false_negative + epsilon),
    }


def save_visualisations(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    count: int,
    threshold: float,
) -> None:
    """Save validation RGB, depth, GT, prediction and overlay panels.

    Reverses RGB normalisation for display. Yellow marks overlap, green missed GT
    branches, and red false foreground. Sets evaluation mode.

    Args:
        model: torch.nn.Module producing branch logits.
        dataset: Dataset yielding ([4, H, W], [1, H, W], key).
        device: torch.device matching model parameters.
        output_dir: pathlib.Path; destination created if needed.
        count: int; nonnegative maximum initial samples to render.
        threshold: float in [0, 1]; inclusive sigmoid cutoff.

    Returns:
        None.

    Raises:
        OSError: If image reading or figure writing fails.
        RuntimeError: If geometry or model execution fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for index in range(min(count, len(dataset))):
            inputs, targets, key = dataset[index]
            probabilities = torch.sigmoid(
                model(inputs.unsqueeze(0).to(device))
            )[0, 0].cpu().numpy()
            prediction = (probabilities >= threshold).astype(np.uint8) * 255
            ground_truth = (targets[0].numpy() >= 0.5).astype(np.uint8) * 255

            # Reverse RGB normalisation for display; model inputs remain unchanged.
            rgb = np.clip(
                (inputs[:3].numpy().transpose(1, 2, 0) * RGB_STD + RGB_MEAN)
                * 255,
                0,
                255,
            ).astype(np.uint8)
            depth = np.clip(inputs[3].numpy() * 255, 0, 255).astype(np.uint8)

            overlay = rgb.astype(np.float32).copy()
            gt_mask = ground_truth > 0
            prediction_mask = prediction > 0

            # Distinguish correct detections, missed branches, and false positives.
            overlap = gt_mask & prediction_mask
            missed = gt_mask & ~prediction_mask
            false_detections = prediction_mask & ~gt_mask
            overlay[overlap] = 0.5 * overlay[overlap] + 0.5 * np.array(
                [255, 255, 0], dtype=np.float32
            )
            overlay[missed] = 0.5 * overlay[missed] + 0.5 * np.array(
                [0, 255, 0], dtype=np.float32
            )
            overlay[false_detections] = (
                0.5 * overlay[false_detections]
                + 0.5 * np.array([255, 0, 0], dtype=np.float32)
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            panels = [
                rgb,
                np.repeat(depth[..., None], 3, axis=2),
                np.repeat(ground_truth[..., None], 3, axis=2),
                np.repeat(prediction[..., None], 3, axis=2),
                overlay,
            ]
            titles = ["RGB", "Depth", "Ground truth", "MLP prediction", "Overlay"]

            figure, axes = plt.subplots(1, len(panels), figsize=(18, 4))
            for axis, panel, title in zip(axes, panels, titles):
                axis.imshow(panel)
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()

            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            figure.savefig(output_dir / f"{safe_key}_comparison.png", dpi=130)
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse MLP training options from the process command line.

    Returns:
        argparse.Namespace: Dataset, output, model and training settings.

    Raises:
        SystemExit: If --help is requested or required arguments are
            missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Train a pixel-wise MLP for RGB-D branch segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root directory containing organised RGB, depth, and GT images.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/pixel_mlp_crop_trainval"),
        help="Directory for the checkpoint, metrics, history, and visualisations.",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Input height and width.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--val-fraction", type=float, default=0.20,
        help="Fraction of samples assigned to validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mask-threshold", type=int, default=127,
        help="GT intensities above this value are labelled as branch.",
    )
    parser.add_argument(
        "--early-stopping", type=int, default=12,
        help="Stop after this many epochs without improved validation Dice; 0 disables it.",
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--visualise-count", type=int, default=30,
        help="Maximum number of validation examples to save as comparison figures.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold used for validation metrics and visualisations.",
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[64, 32, 16],
        help="Number of units in each MLP hidden layer.",
    )
    return parser.parse_args()


def main() -> None:
    """Train, select by validation Dice and export checkpoint and metrics.

    Uses a seeded shuffled sample split, not a held-out tree split. Writes history.csv,
    val_metrics.csv and visualisations. Checkpoint selection uses the same validation
    set. Use positive epochs and validation fraction strictly between 0 and 1.

    Returns:
        None.

    Raises:
        SystemExit: If CLI parsing fails or --help is requested.
        FileNotFoundError: If dataset root is absent.
        RuntimeError: If pairing, split or model execution fails.
        OSError: If reading inputs or writing outputs fails.
        IndexError: If zero epochs leave history empty at CSV export.
    """
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Model: fully connected pixel-wise MLP (no convolutional layers)")
    print("Input: 4 channels (RGB + grayscale depth)")
    print(f"Hidden layers: {args.hidden_dims}")
    print("Loss: weighted BCEWithLogitsLoss")
    print("Optimiser: Adam")

    samples = build_samples(args.data_root)
    random.Random(args.seed).shuffle(samples)

    n_val = max(1, round(len(samples) * args.val_fraction))
    n_train = len(samples) - n_val
    if n_train < 1:
        raise RuntimeError("Not enough samples for train/validation split.")

    train_dataset = BranchDataset(
        samples[:n_train], args.input_size, args.mask_threshold
    )
    val_dataset = BranchDataset(
        samples[n_train:], args.input_size, args.mask_threshold
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = PixelMLP(hidden_dims=tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dice = -1.0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_time = time.time()
    checkpoint_path = args.output_dir / "pixel_mlp_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = bce_loss(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        validation_metrics = evaluate(model, val_loader, device, args.threshold)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"Dice={validation_metrics['dice']:.4f} "
            f"IoU={validation_metrics['iou']:.4f} "
            f"P={validation_metrics['precision']:.4f} "
            f"R={validation_metrics['recall']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{name}": value for name, value in validation_metrics.items()},
            }
        )

        # Select checkpoints by validation Dice, the primary model-selection metric.
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": "PixelMLP",
                    "model_state_dict": best_state,
                    "input_size": args.input_size,
                    "input_channels": ["R", "G", "B", "depth"],
                    "hidden_dims": args.hidden_dims,
                    "mask_threshold": args.mask_threshold,
                    "best_val_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping > 0
            and epochs_without_improvement >= args.early_stopping
        ):
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Training time: {(time.time() - start_time) / 60:.1f} min")
    visualisation_dir = args.output_dir / "visualisations"
    save_visualisations(
        model,
        val_dataset,
        device,
        visualisation_dir,
        args.visualise_count,
        args.threshold,
    )

    final_metrics = evaluate(model, val_loader, device, args.threshold)
    print(
        f"FINAL VAL Dice={final_metrics['dice']:.4f} "
        f"IoU={final_metrics['iou']:.4f} "
        f"Precision={final_metrics['precision']:.4f} "
        f"Recall={final_metrics['recall']:.4f}"
    )

    # Save a compact final summary separately from the per-epoch training history.
    with (args.output_dir / "val_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(final_metrics.keys()))
        writer.writeheader()
        writer.writerow(final_metrics)

    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Visualisations: {visualisation_dir.resolve()}")


if __name__ == "__main__":
    main()
