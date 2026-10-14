#!/usr/bin/env python3
"""Train and evaluate a pixel-wise MLP for RGB-D branch segmentation.

This script trains a fully connected multilayer perceptron (MLP) to classify
individual pixels as branch or background using RGB and grayscale-depth input.
Unlike a convolutional model, the MLP processes each pixel independently and
does not explicitly learn spatial relationships between neighbouring pixels.

The script pairs summer RGB, depth, and ground-truth (GT) PNG files by their
shared timestamp identifier, creates a reproducible train/validation split,
trains the model with weighted binary cross-entropy, and saves the best
checkpoint, validation metrics, training history, and qualitative comparisons.

Expected input:
    - RGB image: three colour channels.
    - Depth image: one grayscale channel.
    - GT mask: one grayscale channel, where values above ``mask_threshold``
      represent branch pixels.

Example:
    python train_pixel_mlp.py \\
        --data-root /path/to/data/Organised \\
        --output-dir outputs/pixel_mlp \\
        --epochs 60 --batch-size 8

Notes:
    RGB images are centre-cropped to a square, resized with bilinear
    interpolation, and normalised using ImageNet RGB statistics. Depth images
    are centre-cropped, resized with nearest-neighbour interpolation, and
    scaled to [0, 1]. GT masks are expected to already match ``input_size``.
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
    """Seed random number generators for more reproducible experiments.

    Args:
        seed: Seed applied to Python, NumPy, and PyTorch random generators.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sample_key(path: Path, modality: str) -> str:
    """Return a modality-independent identifier for an image filename.

    Args:
        path: Path to an RGB, depth, or GT image.
        modality: One of ``"rgb"``, ``"depth"``, or ``"gt"``.

    Returns:
        The filename stem with its modality token replaced by ``_TIME-``.

    Raises:
        KeyError: If ``modality`` is not a supported modality.
    """
    return MODALITY_PATTERNS[modality].sub("_TIME-", path.stem, count=1)


def list_modality(root: Path, modality: str) -> list[Path]:
    """Find summer PNG files belonging to a specified image modality.

    Args:
        root: Directory recursively searched for PNG files.
        modality: One of ``"rgb"``, ``"depth"``, or ``"gt"``.

    Returns:
        Matching file paths in sorted order.

    Raises:
        KeyError: If ``modality`` is not a supported modality.
    """
    pattern = MODALITY_PATTERNS[modality]
    return sorted(
        path
        for path in root.rglob("*.png")
        if pattern.search(path.stem) and "summer" in path.stem.lower()
    )


def map_unique(paths: list[Path], modality: str) -> dict[str, Path]:
    """Map sample identifiers to paths and reject duplicate identifiers.

    Args:
        paths: Image paths for a single modality.
        modality: Modality name used in error messages and key generation.

    Returns:
        Dictionary mapping each sample identifier to its image path.

    Raises:
        RuntimeError: If two files in the modality map to the same identifier.
        KeyError: If ``modality`` is not a supported modality.
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
    """Pair summer RGB, depth, and GT images using their shared identifiers.

    Args:
        root: Root directory containing the organised dataset.

    Returns:
        Sorted sample dictionaries containing ``key``, ``rgb``, ``depth``,
        and ``gt`` entries.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        RuntimeError: If no complete RGB/depth/GT triplets are found, or if
            duplicate identifiers occur within a modality.
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
    """Centre-crop a PIL image to its largest square region.

    Args:
        image: Input PIL image.

    Returns:
        A square crop centred within the original image.
    """
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


class BranchDataset(Dataset):
    """Load and preprocess RGB-D inputs with binary branch segmentation masks.

    RGB and depth images are centre-cropped and resized to ``input_size``.
    The GT mask is not resized: its dimensions must already match the requested
    input size, preventing silent changes to the label geometry.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        input_size: int = 256,
        mask_threshold: int = 127,
    ) -> None:
        """Initialise the dataset.

        Args:
            samples: Sample dictionaries returned by :func:`build_samples`.
            input_size: Height and width of each model input.
            mask_threshold: GT intensities strictly above this value are branch
                pixels; all other pixels are treated as background.
        """
        self.samples = samples
        self.input_size = input_size
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        """Return the number of paired samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load one sample and return its RGB-D tensor and binary target.

        Args:
            index: Index of the sample to load.

        Returns:
            A tuple containing the input tensor ``[4, H, W]``, target tensor
            ``[1, H, W]``, and sample identifier.

        Raises:
            RuntimeError: If the GT mask dimensions differ from ``input_size``.
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
    """Classify each pixel independently using its four RGB-D input values.

    The network consists only of fully connected layers and ReLU activations.
    Spatial dimensions are temporarily flattened into a pixel batch, then
    restored in the output so the model returns one logit per pixel.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
    ) -> None:
        """Build the MLP layers.

        Args:
            in_features: Number of input features per pixel (four for RGB-D).
            hidden_dims: Number of units in each hidden fully connected layer.
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
        """Compute branch logits for an RGB-D batch.

        Args:
            x: Input tensor with shape ``[batch, channels, height, width]``.

        Returns:
            Logit tensor with shape ``[batch, 1, height, width]``.
        """
        batch_size, channels, height, width = x.shape
        pixels = x.permute(0, 2, 3, 1).reshape(-1, channels)
        logits = self.network(pixels)
        return logits.reshape(batch_size, height, width).unsqueeze(1)


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate weighted binary cross-entropy loss from logits.

    Args:
        logits: Raw model predictions with shape matching ``targets``.
        targets: Binary target mask, with values in {0, 1}.

    Returns:
        Scalar weighted binary cross-entropy loss.
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
    """Evaluate segmentation loss and pixel-level metrics over a data loader.

    Metrics are aggregated across all pixels in the loader, rather than averaged
    per image. Predictions are binarised using the supplied probability threshold.

    Args:
        model: Trained segmentation model.
        loader: Data loader containing RGB-D inputs and GT masks.
        device: Device used for model inference.
        threshold: Probability threshold used to classify branch pixels.

    Returns:
        Dictionary containing ``loss``, ``dice``, ``iou``, ``precision``, and
        ``recall``.
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
    """Save side-by-side RGB-D, GT, prediction, and overlay visualisations.

    Overlay colours indicate agreement and error regions:
        - Yellow: predicted branch and GT branch.
        - Green: GT branch missed by the model.
        - Red: predicted branch absent from the GT mask.

    Args:
        model: Trained segmentation model.
        dataset: Dataset to visualise, normally the validation dataset.
        device: Device used for model inference.
        output_dir: Directory where comparison images are written.
        count: Maximum number of samples to visualise.
        threshold: Probability threshold used to binarise predictions.

    Returns:
        None.
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
    """Parse command-line options for training and evaluation.

    Returns:
        Parsed command-line arguments.
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
    """Run dataset preparation, training, evaluation, and artifact export."""
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
