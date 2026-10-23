#!/usr/bin/env python3
"""Train branch segmentation with ASPP, attention and residual decoding.

Preprocessing matches train_unet.py: float32 [B, 4, S, S] RGB-D to [B, 1, S, S] logits.
Uses plain BCE, AdamW (weight decay 1e-4), a validation-Dice plateau scheduler and
paired flips. RGB brightness/contrast perturbation operates after normalisation, leaving
depth unchanged. Architecture intent does not establish improved measured accuracy.

Requires complete summer RGB/depth/branch-GT triplets. Writes the best checkpoint,
history.csv, val_metrics.csv and validation panels under
outputs/improved_resnet34_unet_output by default; named files can be overwritten.

Example:
    python src/train_improved_unet.py --data-root data/Organised --no-pretrained
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


class ConvNormAct(nn.Module):
    """Extract features with convolution, batch normalisation and ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        """Build a padded feature convolution.

        Args:
            in_ch: int; positive input channels.
            out_ch: int; positive output channels.
            kernel_size: int; positive kernel side, normally odd.
            dilation: int; positive dilation rate.

        Returns:
            None.
        """
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        """Transform a batch of features.

        Args:
            x: torch.Tensor; [B, input_channels, H, W].

        Returns:
            torch.Tensor: [B, output_channels, H, W] features for odd kernels; not
                branch logits.

        Raises:
            RuntimeError: If channels, batch-normalisation input or device is
                incompatible.
        """
        return self.block(x)


class ResidualDecoderBlock(nn.Module):
    """Fuse resized decoder and skip features through a residual path."""
    def __init__(self, in_ch, skip_ch, out_ch):
        """Construct convolutions and a channel-matching shortcut.

        Args:
            in_ch: int; positive decoder channels.
            skip_ch: int; skip channels, or 0 when no skip will be provided.
            out_ch: int; positive output channels.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = ConvNormAct(in_ch + skip_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch + skip_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, skip=None):
        """Resize to the skip grid, concatenate and add a residual projection.

        Args:
            x: torch.Tensor; [B, decoder_channels, H, W].
            skip: torch.Tensor or None; [B, skip_channels, Hs, Ws] encoder features.

        Returns:
            torch.Tensor: [B, output_channels, Hs, Ws] with skip; otherwise retains
                input grid.

        Raises:
            RuntimeError: If concatenated channels or devices are incompatible.
        """
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        # Projection makes concatenated skip/context channels compatible with addition.
        residual = self.shortcut(x)
        return self.relu(self.conv2(self.conv1(x)) + residual)


class AttentionGate(nn.Module):
    """Weight encoder skip features using decoder context."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        """Construct context projections and scalar spatial attention.

        Args:
            gate_ch: int; positive context channels.
            skip_ch: int; positive encoder channels.
            inter_ch: int; positive intermediate channels.

        Returns:
            None.
        """
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.score = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())
    def forward(self, gate, skip):
        """Apply sigmoid attention while retaining the skip shape.

        Args:
            gate: torch.Tensor; [B, gate_channels, Hg, Wg] context.
            skip: torch.Tensor; [B, skip_channels, Hs, Ws] encoder features.

        Returns:
            torch.Tensor: Weighted skip [B, skip_channels, Hs, Ws], not a
                one-channel prediction.

        Raises:
            RuntimeError: If channels, batch dimensions or devices are incompatible.
        """
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.score(self.gate(gate) + self.skip(skip))
        return skip * alpha


class ASPP(nn.Module):
    """Combine parallel convolutions at dilation rates 1, 2, 4 and 6.

    The rate-1 branch uses a 1x1 kernel. Projection fuses concatenated features; dropout
    is active only in training mode.
    """
    def __init__(self, in_ch, out_ch):
        """Construct four branches and a context projection.

        Args:
            in_ch: int; positive input channels.
            out_ch: int; output channels, at least 4; branch width is
                floor(output_channels/4).

        Returns:
            None.
        """
        super().__init__()
        branch_ch = out_ch // 4
        self.branches = nn.ModuleList([
            ConvNormAct(in_ch, branch_ch, kernel_size=1),
            ConvNormAct(in_ch, branch_ch, dilation=2),
            ConvNormAct(in_ch, branch_ch, dilation=4),
            ConvNormAct(in_ch, branch_ch, dilation=6),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
        )
    def forward(self, x):
        """Fuse multiscale features without changing the spatial grid.

        Args:
            x: torch.Tensor; [B, input_channels, H, W].

        Returns:
            torch.Tensor: [B, output_channels, H, W] context features, not branch
                logits.

        Raises:
            RuntimeError: If convolution or batch-normalisation input is
                incompatible.
        """
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ImprovedResNet34UNet(nn.Module):
    """Predict branch logits with a four-channel ResNet-34 U-Net.

    ASPP supplies multiscale context, attention weights skip features, and residual
    blocks fuse them. With pretraining, RGB filters are retained and their mean
    initialises depth.
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
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        with torch.no_grad():
            if pretrained:
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        backbone.conv1 = new_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 @ 1/2
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1  # 64 @ 1/4
        self.enc2 = backbone.layer2  # 128 @ 1/8
        self.enc3 = backbone.layer3  # 256 @ 1/16
        self.enc4 = backbone.layer4  # 512 @ 1/32
        self.context = ASPP(512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(192, 128, 64)
        self.att1 = AttentionGate(128, 64, 32)
        self.att0 = AttentionGate(96, 64, 32)
        self.dec3 = ResidualDecoderBlock(256, 256, 192)  # 1/16
        self.dec2 = ResidualDecoderBlock(192, 128, 128)  # 1/8
        self.dec1 = ResidualDecoderBlock(128, 64, 96)    # 1/4
        self.dec0 = ResidualDecoderBlock(96, 64, 64)     # 1/2
        self.refine = nn.Sequential(
            ConvNormAct(64, 48),
            nn.Dropout2d(0.05),
            ConvNormAct(48, 32),
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)
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
        s0 = self.stem(x)
        e1 = self.enc1(self.maxpool(s0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.context(e4)
        skip3 = self.att3(z, e3)
        d3 = self.dec3(z, skip3)
        skip2 = self.att2(d3, e2)
        d2 = self.dec2(d3, skip2)
        skip1 = self.att1(d2, e1)
        d1 = self.dec1(d2, skip1)
        skip0 = self.att0(d1, s0)
        d0 = self.dec0(d1, skip0)
        d0 = F.interpolate(d0, size=input_size, mode="bilinear", align_corners=False)
        d0 = self.refine(d0)
        return self.out(d0)


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
        default=Path("outputs/improved_resnet34_unet_output"),
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--early-stopping", type=int, default=15)
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
    print("Model: improved ResNet-34 U-Net (ASPP + attention skips + residual decoder)")
    print("Input: 4 channels (RGB + one greyscale depth channel)")
    print("Loss: plain BCEWithLogitsLoss")
    print("Optimiser: AdamW")
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
    model = ImprovedResNet34UNet(in_channels=4, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    best_dice, best_state, no_improve = -1.0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            # Paired, label-preserving augmentation applied only to training batches.
            if model.training:
                if torch.rand((), device=device) < 0.5:
                    x, y = torch.flip(x, dims=[3]), torch.flip(y, dims=[3])
                if torch.rand((), device=device) < 0.20:
                    x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
                # Perturb already-standardised RGB; encoded depth values stay unchanged.
                rgb = x[:, :3]
                contrast = 0.90 + 0.20 * torch.rand((x.size(0), 1, 1, 1), device=device)
                brightness = -0.08 + 0.16 * torch.rand((x.size(0), 1, 1, 1), device=device)
                x = torch.cat([rgb * contrast + brightness, x[:, 3:4]], dim=1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / max(len(train_ds), 1)
        val = evaluate(model, val_loader, device, args.threshold)
        scheduler.step(val["dice"])
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
                "model_name": "ImprovedResNet34UNet",
                "model_state_dict": best_state,
                "input_size": args.input_size,
                "input_channels": ["R", "G", "B", "depth"],
                "pretrained_encoder": not args.no_pretrained,
                "mask_threshold": args.mask_threshold,
                "best_val_metrics": val,
                "architecture": "ResNet34 + ASPP + attention skips + residual decoder",
            }, args.output_dir / "improved_resnet34_unet_best.pth")
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
    print(f"Checkpoint: {(args.output_dir / 'improved_resnet34_unet_best.pth').resolve()}")
    print(f"Visualisations: {(args.output_dir / 'visualisations').resolve()}")

if __name__ == "__main__":
    main()
