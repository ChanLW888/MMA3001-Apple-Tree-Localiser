train_improved_unet
===================

.. py:module:: train_improved_unet

Source: ``src/train_improved_unet.py``.

Statically extracted signatures and existing docstrings; no module execution.

Train branch segmentation with ASPP, attention and residual decoding.

Preprocessing matches train_unet.py: float32 [B, 4, S, S] RGB-D to [B, 1, S, S] logits.
Uses plain BCE, AdamW (weight decay 1e-4), a validation-Dice plateau scheduler and
paired flips. RGB brightness/contrast perturbation operates after normalisation, leaving
depth unchanged. Architecture intent does not establish improved measured accuracy.

.. rubric:: Example

python src/train_improved_unet.py --data-root data/Organised --no-pretrained


.. py:function:: seed_everything(seed)

   Seed Python, NumPy and PyTorch generators.
   
   CUDA generators are seeded when available. Seeding does not guarantee deterministic
   GPU kernels.
   
   :param seed: int; nonnegative NumPy-compatible random seed.
   
   :returns: None.
   
   :raises ValueError: If NumPy rejects the seed.

.. py:function:: get_key(path, modality)

   Build a modality-independent filename identifier.
   
   :param path: pathlib.Path to an image; stem retains tree, season and time.
   :param modality: str; one of "rgb", "depth", or "gt".
   
   :returns: Stem with the first modality token replaced by "_TIME-".
   :rtype: str
   
   :raises KeyError: If modality is unsupported.

.. py:function:: list_modality(root, modality)

   Find summer PNGs for one modality recursively.
   
   :param root: pathlib.Path to the dataset directory.
   :param modality: str; "rgb", "depth", or "gt".
   
   :returns: Sorted matching paths with summer in the stem.
   :rtype: list[pathlib.Path]
   
   :raises KeyError: If modality is unsupported.

.. py:function:: map_unique(paths, modality)

   Index paths while rejecting ambiguous sample identifiers.
   
   :param paths: list[pathlib.Path] for one modality.
   :param modality: str; "rgb", "depth", or "gt".
   
   :returns: Identifiers mapped to unique files.
   :rtype: dict[str, pathlib.Path]
   
   :raises RuntimeError: If paths produce duplicate identifiers.
   :raises KeyError: If modality is unsupported.

.. py:function:: build_samples(root)

   Pair complete summer RGB, depth and branch-GT triplets.
   
   Incomplete triplets are omitted. Filename agreement does not verify spatial
   alignment.
   
   :param root: pathlib.Path; existing organised dataset root.
   
   :returns:
   
             Sorted records with key, rgb, depth and gt entries; paths are
                 pathlib.Path objects.
   :rtype: list[dict]
   
   :raises FileNotFoundError: If root does not exist.
   :raises RuntimeError: If identifiers repeat or no complete triplets exist.

.. py:function:: centre_crop_square(img)

   Crop to the largest centred square without resampling.
   
   :param img: PIL.Image.Image in any mode.
   
   :returns: Square crop of side min(width, height).
   :rtype: PIL.Image.Image

.. py:class:: BranchDataset(Dataset)

   Load aligned RGB-D inputs and binary branch targets.
   
   RGB and depth are centre-cropped and resized. GT is neither cropped nor resized; it
   must already describe that field of view at input_size. Channel order is R, G, B,
   then grayscale depth. Depth is image intensity, not calibrated metres.

.. py:method:: BranchDataset.__init__(self, samples, input_size=256, mask_threshold=127)

   Store paired samples and preprocessing settings.
   
   :param samples: list[dict] from build_samples; rgb, depth, gt and key entries.
   :param input_size: int; positive square side in pixels, default 256.
   :param mask_threshold: int; grayscale cutoff, normally [0, 255]; GT > cutoff is
                          branch.
   
   :returns: None.

.. py:method:: BranchDataset.__len__(self)

   Return the number of complete samples.
   
   :returns: Number of stored records.
   :rtype: int

.. py:method:: BranchDataset.__getitem__(self, index)

   Read and preprocess one RGB-D and branch-GT sample.
   
   RGB is bilinearly resized, /255 then ImageNet standardised. Depth is converted
   to L, nearest-neighbour resized and /255. Targets use GT > mask_threshold; 1
   denotes branch and 0 background.
   
   :param index: int; Python list index into samples.
   
   :returns:
   
             Float32 input [4, S, S], binary
                 float32 target [1, S, S], and identifier; S is input_size.
   :rtype: tuple[torch.Tensor, torch.Tensor, str]
   
   :raises IndexError: If index is outside samples.
   :raises KeyError: If a sample lacks a required entry.
   :raises OSError: If an input cannot be opened or decoded.
   :raises RuntimeError: If GT size differs from input_size.

.. py:class:: ConvNormAct(nn.Module)

   Extract features with convolution, batch normalisation and ReLU.

.. py:method:: ConvNormAct.__init__(self, in_ch, out_ch, kernel_size=3, dilation=1)

   Build a padded feature convolution.
   
   :param in_ch: int; positive input channels.
   :param out_ch: int; positive output channels.
   :param kernel_size: int; positive kernel side, normally odd.
   :param dilation: int; positive dilation rate.
   
   :returns: None.

.. py:method:: ConvNormAct.forward(self, x)

   Transform a batch of features.
   
   :param x: torch.Tensor; [B, input_channels, H, W].
   
   :returns:
   
             [B, output_channels, H, W] features for odd kernels; not
                 branch logits.
   :rtype: torch.Tensor
   
   :raises RuntimeError: If channels, batch-normalisation input or device is
       incompatible.

.. py:class:: ResidualDecoderBlock(nn.Module)

   Fuse resized decoder and skip features through a residual path.

.. py:method:: ResidualDecoderBlock.__init__(self, in_ch, skip_ch, out_ch)

   Construct convolutions and a channel-matching shortcut.
   
   :param in_ch: int; positive decoder channels.
   :param skip_ch: int; skip channels, or 0 when no skip will be provided.
   :param out_ch: int; positive output channels.
   
   :returns: None.

.. py:method:: ResidualDecoderBlock.forward(self, x, skip=None)

   Resize to the skip grid, concatenate and add a residual projection.
   
   :param x: torch.Tensor; [B, decoder_channels, H, W].
   :param skip: torch.Tensor or None; [B, skip_channels, Hs, Ws] encoder features.
   
   :returns:
   
             [B, output_channels, Hs, Ws] with skip; otherwise retains
                 input grid.
   :rtype: torch.Tensor
   
   :raises RuntimeError: If concatenated channels or devices are incompatible.

.. py:class:: AttentionGate(nn.Module)

   Weight encoder skip features using decoder context.

.. py:method:: AttentionGate.__init__(self, gate_ch, skip_ch, inter_ch)

   Construct context projections and scalar spatial attention.
   
   :param gate_ch: int; positive context channels.
   :param skip_ch: int; positive encoder channels.
   :param inter_ch: int; positive intermediate channels.
   
   :returns: None.

.. py:method:: AttentionGate.forward(self, gate, skip)

   Apply sigmoid attention while retaining the skip shape.
   
   :param gate: torch.Tensor; [B, gate_channels, Hg, Wg] context.
   :param skip: torch.Tensor; [B, skip_channels, Hs, Ws] encoder features.
   
   :returns:
   
             Weighted skip [B, skip_channels, Hs, Ws], not a
                 one-channel prediction.
   :rtype: torch.Tensor
   
   :raises RuntimeError: If channels, batch dimensions or devices are incompatible.

.. py:class:: ASPP(nn.Module)

   Combine parallel convolutions at dilation rates 1, 2, 4 and 6.
   
   The rate-1 branch uses a 1x1 kernel. Projection fuses concatenated features; dropout
   is active only in training mode.

.. py:method:: ASPP.__init__(self, in_ch, out_ch)

   Construct four branches and a context projection.
   
   :param in_ch: int; positive input channels.
   :param out_ch: int; output channels, at least 4; branch width is
                  floor(output_channels/4).
   
   :returns: None.

.. py:method:: ASPP.forward(self, x)

   Fuse multiscale features without changing the spatial grid.
   
   :param x: torch.Tensor; [B, input_channels, H, W].
   
   :returns:
   
             [B, output_channels, H, W] context features, not branch
                 logits.
   :rtype: torch.Tensor
   
   :raises RuntimeError: If convolution or batch-normalisation input is
       incompatible.

.. py:class:: ImprovedResNet34UNet(nn.Module)

   Predict branch logits with a four-channel ResNet-34 U-Net.
   
   ASPP supplies multiscale context, attention weights skip features, and residual
   blocks fuse them. With pretraining, RGB filters are retained and their mean
   initialises depth.

.. py:method:: ImprovedResNet34UNet.__init__(self, in_channels=4, pretrained=True)

   Construct encoder and decoder layers.
   
   :param in_channels: int; intended value 4; pretrained initialisation assumes
                       RGB-D.
   :param pretrained: bool; request default torchvision ResNet-34 weights when
                      True.
   
   :returns: None.
   
   :raises OSError: If pretrained weights cannot be read or retrieved.
   :raises RuntimeError: If layer dimensions or weights are incompatible.

.. py:method:: ImprovedResNet34UNet.forward(self, x)

   Decode RGB-D features into full-resolution branch logits.
   
   :param x: torch.Tensor; floating [B, 4, H, W] batch using training
             preprocessing.
   
   :returns:
   
             Raw [B, 1, H, W] logits; sigmoid converts to
                 probabilities.
   :rtype: torch.Tensor
   
   :raises RuntimeError: If channels, spatial size or dtype/device is incompatible.

.. py:function:: bce_loss(logits, targets)

   Compute binary cross-entropy directly from branch logits.
   
   Uses plain unweighted BCE; sigmoid is internal to the numerically stable loss.
   
   :param logits: torch.Tensor; floating [B, 1, H, W] raw scores.
   :param targets: torch.Tensor; same shape/device, floating labels in {0, 1}.
   
   :returns: Scalar mean loss.
   :rtype: torch.Tensor
   
   :raises ValueError: If target and logit sizes differ.
   :raises RuntimeError: If tensor types or devices are incompatible.

.. py:function:: evaluate(model, loader, device, threshold)

   Aggregate validation loss and foreground pixel metrics.
   
   Sets evaluation mode and disables gradients. Uses sigmoid(logits) >= threshold and
   targets >=0.5. Epsilon 1e-8 stabilises ratios; empty foreground scores zero.
   
   :param model: torch.nn.Module mapping [B, 4, H, W] to [B, 1, H, W] logits.
   :param loader: DataLoader yielding input, target and identifier batches.
   :param device: torch.device; model must already be on it.
   :param threshold: float; inclusive sigmoid cutoff, normally [0, 1].
   
   :returns:
   
             Sample-weighted loss and pixel-micro dice, iou, precision
                 and recall.
   :rtype: dict[str, float]
   
   :raises OSError: If dataset inputs cannot be read.
   :raises RuntimeError: If dataset geometry or model execution fails.

.. py:function:: save_visualisations(model, dataset, device, output_dir, count, threshold)

   Save validation RGB, depth, GT, prediction and overlay panels.
   
   Reverses RGB normalisation for display. Yellow marks overlap, green missed GT
   branches, and red false foreground. Sets evaluation mode.
   
   :param model: torch.nn.Module producing branch logits.
   :param dataset: Dataset yielding ([4, H, W], [1, H, W], key).
   :param device: torch.device matching model parameters.
   :param output_dir: pathlib.Path; destination created if needed.
   :param count: int; nonnegative maximum initial samples to render.
   :param threshold: float in [0, 1]; inclusive sigmoid cutoff.
   
   :returns: None.
   
   :raises OSError: If image reading or figure writing fails.
   :raises RuntimeError: If geometry or model execution fails.

.. py:function:: main()

   Train, select by validation Dice and export checkpoint and metrics.
   
   Uses a seeded shuffled sample split, not a held-out tree split. Writes history.csv,
   val_metrics.csv and visualisations. Checkpoint selection uses the same validation
   set. Use positive epochs and validation fraction strictly between 0 and 1.
   
   :returns: None.
   
   :raises SystemExit: If CLI parsing fails or --help is requested.
   :raises FileNotFoundError: If dataset root is absent.
   :raises RuntimeError: If pairing, split or model execution fails.
   :raises OSError: If reading inputs or writing outputs fails.
   :raises IndexError: If zero epochs leave history empty at CSV export.

