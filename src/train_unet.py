#!/usr/bin/env python3
"""
Train a ResNet-34 encoder U-Net for binary Y-shaped apple-tree branch segmentation.
Paper-derived architecture:
*\\-* U-Net encoder-decoder with ResNet-34 encoder and skip connections.
*\\-* RGB-D input and one-channel binary mask output.
*\\-* Original paper: TensorFlow, Adam, weighted Dice loss, batch size 8.
This PyTorch adaptation keeps the requested simple BCE loss, Adam optimizer,
batch size 8, and 4 input channels (RGB + grayscale depth).
Preprocessing follows the paper's stated geometry:
640x480 RGB/depth -> centre crop to 480x480 -> resize to 256x256.
IMPORTANT: only use this if your GT mask corresponds to that centred crop.
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
    """Set Python, NumPy, and PyTorch random seeds for reproducible runs.

    Args:
        seed: Integer seed shared across supported random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_key(path, modality):
    """Build a modality-independent identifier from an image filename.

    Args:
        path: Path to an RGB, depth, or ground-truth image.
        modality: Image modality key used to select the filename pattern.

    Returns:
        Filename stem with the modality token replaced by ``_TIME-``."""
    return PATTERNS[modality].sub("_TIME-", path.stem, count=1)


def list_modality(root, modality):
    """Find summer PNG files belonging to one dataset modality.

    Args:
        root: Directory recursively searched for images.
        modality: One of ``rgb``, ``depth``, or ``gt``.

    Returns:
        Sorted paths matching the requested modality."""
    pattern = PATTERNS[modality]
    return sorted(
        p for p in root.rglob("*.png")
        if pattern.search(p.stem) and "summer" in p.stem.lower()
    )


def map_unique(paths, modality):
    """Index image paths by their shared sample identifier.

    Args:
        paths: Paths belonging to one modality.
        modality: Modality name used to normalize each filename.

    Returns:
        Mapping from sample identifier to image path.

    Raises:
        RuntimeError: If multiple files share the same identifier."""
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
    """Pair RGB, depth, and ground-truth files into complete samples.

    Args:
        root: Dataset directory containing modality-specific PNG files.

    Returns:
        Samples sorted by their shared identifier.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        RuntimeError: If no complete triplets are found or identifiers repeat."""
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
    """Centre-crop an image to its largest inscribed square.

    Args:
        img: PIL image to crop.

    Returns:
        Cropped PIL image with equal width and height."""
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


    """Load paired RGB-D images and binary branch masks."""
class BranchDataset(Dataset):
    def __init__(self, samples, input_size=256, mask_threshold=127):
        self.samples = samples
        self.input_size = input_size
        self.mask_threshold = mask_threshold
    def __len__(self):
        """Return the number of paired samples in the dataset.

        Returns:
            Number of RGB-depth-mask triplets."""
        return len(self.samples)
    def __getitem__(self, index):
        """Load and preprocess one RGB-D image and its binary branch mask.

        Args:
            index: Index of the sample to load.

        Returns:
            A tuple containing the four-channel input tensor, one-channel target
            tensor, and shared sample identifier.

        Raises:
            RuntimeError: If the ground-truth mask dimensions are unexpected."""
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
        # Exactly four input channels: RGB + one greyscale depth channel.
        # Concatenate RGB and depth as channels: [4, height, width].
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
    """
    U-Net with ResNet-34 encoder.
    Encoder stages use ResNet-34's stem, layer1, layer2, layer3, layer4.
    Decoder upsamples and concatenates encoder skip features at each scale.
    Output is a single-channel logit map; apply sigmoid only for inference.
    """
    def __init__(self, in_channels=4, pretrained=True):
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
        # Decoder channel sizes are chosen to align with the paper's U-Net diagram.
        self.up4 = self._decoder_block(512, 256, 256)  # /16
        self.up3 = self._decoder_block(256 + 256, 128, 128)  # /16
        self.up2 = self._decoder_block(128 + 128, 64, 64)  # /4
        self.up1 = self._decoder_block(64 + 64, 64, 64)  # /2
        self.up0 = self._decoder_block(64 + 64, 32, 32)  # /1
        self.out = nn.Conv2d(32, 1, kernel_size=1)
    @staticmethod
    def _decoder_block(in_ch, mid_ch, out_ch):
        """Create a two-convolution decoder block with batch normalisation.

        Args:
            in_ch: Number of input feature channels.
            mid_ch: Number of channels in the first convolution.
            out_ch: Number of output feature channels.

        Returns:
            Sequential convolutional decoder block."""
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
        """Resize a feature map to the spatial dimensions of a reference tensor.

        Args:
            x: Feature map to resize.
            reference: Tensor whose height and width define the target size.

        Returns:
            Bilinearly interpolated feature map."""
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    def forward(self, x):
        """Run the input through the network and return a one-channel logit map.

        Args:
            x: Input tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Raw branch-segmentation logits shaped ``[batch, 1, height, width]``."""
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
    """Calculate binary cross-entropy directly from logits.

    Args:
        logits: Unnormalised model predictions.
        targets: Binary ground-truth masks with matching dimensions.

    Returns:
        Scalar binary cross-entropy loss."""
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()


def evaluate(model, loader, device, threshold):
    """Evaluate segmentation loss and thresholded pixel-level metrics.

    Args:
        model: Trained segmentation model.
        loader: DataLoader providing inputs and target masks.
        device: Device on which inference is performed.
        threshold: Probability threshold used to classify branch pixels.

    Returns:
        Dictionary containing loss, Dice, IoU, precision, and recall."""
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
    """Save RGB, depth, target, prediction, and overlay comparison panels.

    Args:
        model: Model used to generate predictions.
        dataset: Dataset to visualise.
        device: Device used for inference.
        output_dir: Destination directory for comparison images.
        count: Maximum number of samples to visualise.
        threshold: Probability threshold used to binarise predictions."""
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
    """Parse command-line arguments and run model training and validation."""
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
