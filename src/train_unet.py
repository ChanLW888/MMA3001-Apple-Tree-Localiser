#!/usr/bin/env python3
"""Train a four-channel ResNet-34 U-Net for branch segmentation.

Centre-crops RGB/depth and resizes to input_size (default 256). RGB uses ImageNet
normalisation; grayscale depth is /255, not metres. GT must already match the square
grid. Produces raw [B, 1, H, W] branch logits. Uses plain BCE, Adam and validation-Dice
checkpoint selection. Pretraining may retrieve uncached weights.

Writes resnet34_unet_best.pth, history.csv, val_metrics.csv and visualisations
under outputs/resnet34_unet_crop_trainval_output by default, overwriting names.

Example:
    python src/train_unet.py --data-root data/Organised --no-pretrained
"""
import argparse
import copy
import csv
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights



# Filename tokens used to match RGB, depth, and ground-truth files.
PATTERNS = {
    "rgb": re.compile(r"_RGB-", re.IGNORECASE),
    "depth": re.compile(r"_D-", re.IGNORECASE),
    "gt": re.compile(r"_GT-", re.IGNORECASE),
}


def seed_everything(seed):
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


def get_key(path, modality):
    """Build a modality-independent filename identifier.

    Args:
        path: pathlib.Path to an image; stem retains tree, season and time.
        modality: str; one of "rgb", "depth", or "gt".

    Returns:
        str: Stem with the first modality token replaced by "_TIME-".

    Raises:
        KeyError: If modality is unsupported.
    """
    return PATTERNS[modality].sub("_TIME-", path.stem, count=1)


def list_modality(root, modality):
    """Find summer PNGs for one modality recursively.

    Args:
        root: pathlib.Path to the dataset directory.
        modality: str; "rgb", "depth", or "gt".

    Returns:
        list[pathlib.Path]: Sorted matching paths with summer in the stem.

    Raises:
        KeyError: If modality is unsupported.
    """
    pattern = PATTERNS[modality]
    return sorted(
        p for p in root.rglob("*.png")
        if pattern.search(p.stem) and "summer" in p.stem.lower()
    )


def map_unique(paths, modality):
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
    result = {}
    for path in paths:
        key = get_key(path, modality)
        if key in result:
            raise RuntimeError(
                f"Duplicate {modality} identifier {key}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def build_samples(root):
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
    samples = []
    for key, rgb in rgb_map.items():
        if key in depth_map and key in gt_map:
            samples.append({
                "key": key, "rgb": rgb,
                "depth": depth_map[key], "gt": gt_map[key]
            })
    print("\n========== DATASET PAIRING ==========")
    print(f"RGB={len(rgb_map)}, depth={len(depth_map)}, GT={len(gt_map)}")
    print(f"Complete triplets={len(samples)}")
    if not samples:
        raise RuntimeError("No complete RGB/depth/GT triplets found.")
    return sorted(samples, key=lambda s: s["key"])


def centre_crop_square(img):
    """Crop to the largest centred square without resampling.

    Args:
        img: PIL.Image.Image in any mode.

    Returns:
        PIL.Image.Image: Square crop of side min(width, height).
    """
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


class BranchDataset(Dataset):
    """Load aligned RGB-D inputs and binary branch targets.

    RGB and depth are centre-cropped and resized. GT is neither cropped nor resized; it
    must already describe that field of view at input_size. Channel order is R, G, B,
    then grayscale depth. Depth is image intensity, not calibrated metres.
    """
    def __init__(self, samples, input_size=256, mask_threshold=127):
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
    def __len__(self):
        """Return the number of complete samples.

        Returns:
            int: Number of stored records.
        """
        return len(self.samples)
    def __getitem__(self, index):
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
        rgb_img = centre_crop_square(Image.open(sample["rgb"]).convert("RGB"))
        rgb_img = rgb_img.resize(size, Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
        # ImageNet normalisation for the pretrained ResNet-34 RGB channels.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        depth_img = centre_crop_square(Image.open(sample["depth"]).convert("L"))
        depth_img = depth_img.resize(size, Image.Resampling.NEAREST)
        depth = np.asarray(depth_img, dtype=np.float32) / 255.0
        # Preserve the checkpoint channel contract: normalised RGB, then depth/255.
        x = np.concatenate([rgb.transpose(2, 0, 1), depth[None]], axis=0)
        gt_img = Image.open(sample["gt"]).convert("L")
        if gt_img.size != size:
            raise RuntimeError(
                f"GT size {gt_img.size} != expected {size} for {sample['key']}. "
                "The GT must be 256x256 or you must explicitly change preprocessing."
            )
        gt = np.asarray(gt_img, dtype=np.uint8)
        # Convert the grayscale annotation into a binary, single-channel target.
        y = (gt > self.mask_threshold).astype(np.float32)[None]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y), sample["key"]


class ResNet34UNet(nn.Module):
    """Predict branch logits with a four-channel ResNet-34 U-Net.

    Encoder skip features restore spatial detail during decoding. With pretraining, RGB
    filters are retained and their mean initialises depth.
    """
    def __init__(self, in_channels=4, pretrained=True):
        """Construct encoder and decoder layers.

        Args:
            in_channels: int; intended value 4; pretrained initialisation assumes
                RGB-D.
            pretrained: bool; request default torchvision ResNet-34 weights when
                True.

        Returns:
            None.

        Raises:
            OSError: If pretrained weights cannot be read or retrieved.
            RuntimeError: If layer dimensions or weights are incompatible.
        """
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        # Modify the first convolution from 3 to 4 channels.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                new_conv.weight[:, :3] = old_conv.weight
                # Initialise depth channel from the mean RGB filters.
                new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        backbone.conv1 = new_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1  # /4, 64
        self.enc2 = backbone.layer2  # /8, 128
        self.enc3 = backbone.layer3  # /16, 256
        self.enc4 = backbone.layer4  # /32, 512
        # Decoder widths match the feature channels concatenated in forward().
        self.up4 = self._decoder_block(512, 256, 256)  # /16
        self.up3 = self._decoder_block(256 + 256, 128, 128)  # /16
        self.up2 = self._decoder_block(128 + 128, 64, 64)  # /8
        self.up1 = self._decoder_block(64 + 64, 64, 64)  # /4
        self.up0 = self._decoder_block(64 + 64, 32, 32)  # /2
        self.out = nn.Conv2d(32, 1, kernel_size=1)
    @staticmethod
    def _decoder_block(in_ch, mid_ch, out_ch):
        """Construct a two-convolution decoder feature block.

        Args:
            in_ch: int; positive input channels.
            mid_ch: int; positive hidden channels.
            out_ch: int; positive output channels.

        Returns:
            torch.nn.Sequential: 3x3 convolution, batch-normalisation and ReLU
                layers preserving spatial size.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    @staticmethod
    def _up_to(x, reference):
        """Match a feature map to an encoder grid before concatenation.

        Args:
            x: torch.Tensor; floating [B, C, H, W] features.
            reference: torch.Tensor; last two dimensions define destination size.

        Returns:
            torch.Tensor: Bilinear [B, C, Href, Wref] features, align_corners=False.

        Raises:
            ValueError: If interpolation receives invalid spatial rank.
        """
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    def forward(self, x):
        """Decode RGB-D features into full-resolution branch logits.

        Args:
            x: torch.Tensor; floating [B, 4, H, W] batch using training
                preprocessing.

        Returns:
            torch.Tensor: Raw [B, 1, H, W] logits; sigmoid converts to
                probabilities.

        Raises:
            RuntimeError: If channels, spatial size or dtype/device is incompatible.
        """
        input_size = x.shape[-2:]
        s0 = self.stem(x)               # /2, 64
        e1 = self.enc1(self.maxpool(s0)) # /4, 64
        e2 = self.enc2(e1)              # /8, 128
        e3 = self.enc3(e2)              # /16, 256
        e4 = self.enc4(e3)              # /32, 512
        d4 = self.up4(self._up_to(e4, e3))          # /16, 256
        d3 = self.up3(torch.cat([d4, e3], dim=1))   # /16, 128
        d3 = self.up2(torch.cat([self._up_to(d3, e2), e2], dim=1))  # /8 -> 64
        d2 = self.up1(torch.cat([self._up_to(d3, e1), e1], dim=1))  # /4 -> 64
        d1 = self.up0(torch.cat([self._up_to(d2, s0), s0], dim=1))  # /2 -> 32
        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.out(out)


def bce_loss(logits, targets):
    """Compute binary cross-entropy directly from branch logits.

    Uses plain unweighted BCE; sigmoid is internal to the numerically stable loss.

    Args:
        logits: torch.Tensor; floating [B, 1, H, W] raw scores.
        targets: torch.Tensor; same shape/device, floating labels in {0, 1}.

    Returns:
        torch.Tensor: Scalar mean loss.

    Raises:
        ValueError: If target and logit sizes differ.
        RuntimeError: If tensor types or devices are incompatible.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()


def evaluate(model, loader, device, threshold):
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
    tp = fp = fn = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += bce_loss(logits, y).item() * x.size(0)
        pred = torch.sigmoid(logits) >= threshold
        target = y >= 0.5
        tp += (pred & target).sum().item()
        fp += (pred & ~target).sum().item()
        fn += (~pred & target).sum().item()
    eps = 1e-8
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def save_visualisations(model, dataset, device, output_dir, count, threshold):
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
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(min(count, len(dataset))):
            x, y, key = dataset[i]
            prob = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            pred = (prob >= threshold).astype(np.uint8) * 255
            gt = (y[0].numpy() >= 0.5).astype(np.uint8) * 255
            rgb = np.clip((x[:3].numpy().transpose(1, 2, 0) * std + mean) * 255, 0, 255).astype(np.uint8)
            depth = np.clip(x[3].numpy() * 255, 0, 255).astype(np.uint8)
            overlay = rgb.astype(np.float32).copy()
            gt_b, pred_b = gt > 0, pred > 0
            overlay[gt_b & pred_b] = 0.5 * overlay[gt_b & pred_b] + 0.5 * np.array([255, 255, 0], np.float32)
            overlay[gt_b & ~pred_b] = 0.5 * overlay[gt_b & ~pred_b] + 0.5 * np.array([0, 255, 0], np.float32)
            overlay[pred_b & ~gt_b] = 0.5 * overlay[pred_b & ~gt_b] + 0.5 * np.array([255, 0, 0], np.float32)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            panels = [
                rgb, np.repeat(depth[..., None], 3, axis=2),
                np.repeat(gt[..., None], 3, axis=2),
                np.repeat(pred[..., None], 3, axis=2), overlay
            ]
            titles = ["RGB", "Depth", "GT", "Prediction", "Overlay"]
            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            for ax, panel, title in zip(axes, panels, titles):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            fig.savefig(output_dir / f"{safe_key}_comparison.png", dpi=130)
            plt.close(fig)


def main():
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/resnet34_unet_crop_trainval_output"),
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--early-stopping", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--visualise-count", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Model: U-Net with ResNet-34 encoder")
    print("Input: 4 channels (RGB + one greyscale depth channel)")
    print("Loss: plain BCEWithLogitsLoss")
    print("Optimiser: Adam")
    print(f"Batch size: {args.batch_size}")
    samples = build_samples(args.data_root)
    random.Random(args.seed).shuffle(samples)
    n = len(samples)
    n_val = max(1, round(n * args.val_fraction))
    n_train = n - n_val
    if n_train < 1:
        raise RuntimeError("Not enough samples for train/validation split.")
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    train_ds = BranchDataset(train_samples, args.input_size, args.mask_threshold)
    val_ds = BranchDataset(val_samples, args.input_size, args.mask_threshold)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    model = ResNet34UNet(in_channels=4, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dice, best_state, no_improve = -1.0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / max(len(train_ds), 1)
        val = evaluate(model, val_loader, device, args.threshold)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"Dice={val['dice']:.4f} IoU={val['iou']:.4f} "
            f"P={val['precision']:.4f} R={val['recall']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "model_name": "ResNet34UNet",
                "model_state_dict": best_state,
                "input_size": args.input_size,
                "input_channels": ["R", "G", "B", "depth"],
                "pretrained_encoder": not args.no_pretrained,
                "mask_threshold": args.mask_threshold,
                "best_val_metrics": val,
            }, args.output_dir / "resnet34_unet_best.pth")
        else:
            no_improve += 1
        if args.early_stopping > 0 and no_improve >= args.early_stopping:
            print(f"Early stopping after {epoch} epochs.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training time: {(time.time() - start) / 60:.1f} min; best val Dice={best_dice:.4f}")
    save_visualisations(model, val_ds, device, args.output_dir / "visualisations",
                        args.visualise_count, args.threshold)
    final_val = evaluate(model, val_loader, device, args.threshold)
    print(f"FINAL VAL Dice={final_val['dice']:.4f} IoU={final_val['iou']:.4f} "
          f"Precision={final_val['precision']:.4f} Recall={final_val['recall']:.4f}")
    with (args.output_dir / "val_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_val.keys()))
        writer.writeheader()
        writer.writerow(final_val)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Checkpoint: {(args.output_dir / 'resnet34_unet_best.pth').resolve()}")
    print(f"Visualisations: {(args.output_dir / 'visualisations').resolve()}")

if __name__ == "__main__":
    main()
