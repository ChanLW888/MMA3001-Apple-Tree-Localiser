winter_inference
================

.. py:module:: winter_inference

Source: ``src/winter_inference.py``.

Statically extracted signatures and existing docstrings; no module execution.

Run qualitative winter RGB-D inference and model-only timing.

Reconstructs training MLP/basic/improved U-Nets without ImageNet weight retrieval, then
strictly loads trusted local checkpoints. Uses training-compatible preprocessing. No
winter GT is required; predictions cannot establish winter accuracy. Timing excludes
preprocessing, transfer, sigmoid and saving.

Requires winter RGB-D pairs and all three local model checkpoints. Writes masks,
overlays, panels, per_image_timings.csv, summary.csv and benchmark_metadata.json
under outputs/winter_inference_benchmark by default. Reruns overwrite matching names.

.. rubric:: Example

See README.md for the invocation supplying all three checkpoints.


.. py:function:: crop_square(im)

   Crop a PIL image to its largest centred square.

   :param im: PIL.Image.Image to crop.

   :returns: Square with side min(width, height).
   :rtype: PIL.Image.Image

.. py:function:: sample_key(p, pat)

   Replace the first modality token, retaining frame identity.

   :param p: pathlib.Path to a winter image.
   :param pat: re.Pattern for RGB or depth token.

   :returns: Stem with first token replaced by _TIME-.
   :rtype: str

.. py:function:: discover(root)

   Find complete winter RGB/depth pairs without requiring GT.

   Reports unmatched RGB images and excludes incomplete pairs. Filenames do not verify
   geometric alignment.

   :param root: pathlib.Path; recursively searched dataset root.

   :returns: Sorted key, rgb and depth records.
   :rtype: list[dict]

   :raises RuntimeError: If duplicate keys occur or no pairs exist.

.. py:function:: preprocess(s, size)

   Apply training-compatible RGB-D preprocessing.

   Centre-crops both modalities; RGB is bilinear, /255 then ImageNet standardised.
   Depth is nearest neighbour, grayscale /255, not metric distance.

   :param s: dict with rgb and depth pathlib.Path entries.
   :param size: int; positive square input side in pixels, normally 256.

   :returns:

             Float32 [4, size,
                 size] input, resized RGB and L-mode depth.
   :rtype: tuple[torch.Tensor, PIL.Image.Image, PIL.Image.Image]

   :raises OSError: If images cannot be decoded.
   :raises ValueError: If resize dimensions are invalid.

.. py:class:: PixelMLP(nn.Module)

   Reconstruct the training pixel MLP for checkpoint inference.

.. py:method:: PixelMLP.__init__(self, hidden_dims=(64, 32, 16))

   Construct four-feature pixel layers with matching checkpoint names.

   :param hidden_dims: tuple[int, ...]; positive hidden widths matching the
                       checkpoint.

   :returns: None.

.. py:method:: PixelMLP.forward(self, x)

   Classify each pixel independently.

   :param x: torch.Tensor; [B, 4, H, W] preprocessed RGB-D.

   :returns: Raw [B, 1, H, W] branch logits.
   :rtype: torch.Tensor

   :raises RuntimeError: If feature count or dtype/device is incompatible.

.. py:function:: decoder_block(in_ch, mid_ch, out_ch)

   Construct a two-convolution decoder feature block.

   :param in_ch: int; positive input channels.
   :param mid_ch: int; positive hidden channels.
   :param out_ch: int; positive output channels.

   :returns:

             3x3 convolution, batch-normalisation and ReLU layers
                 preserving spatial size.
   :rtype: torch.nn.Sequential

.. py:class:: BasicUNet(nn.Module)

   Predict branch logits with a four-channel ResNet-34 U-Net.

   Encoder skip features restore spatial detail during decoding. Layers match training
   checkpoint keys; construction uses weights=None.

.. py:method:: BasicUNet.__init__(self)

   Construct encoder and decoder layers.

   :returns: None.

.. py:method:: BasicUNet.up(x, ref)

   Match a feature map to an encoder grid before concatenation.

   :param x: torch.Tensor; floating [B, C, H, W] features.
   :param ref: torch.Tensor; last two dimensions define destination size.

   :returns: Bilinear [B, C, Href, Wref] features, align_corners=False.
   :rtype: torch.Tensor

   :raises ValueError: If interpolation receives invalid spatial rank.

.. py:method:: BasicUNet.forward(self, x)

   Decode RGB-D features into full-resolution branch logits.

   :param x: torch.Tensor; floating [B, 4, H, W] batch using training
             preprocessing.

   :returns:

             Raw [B, 1, H, W] logits; sigmoid converts to
                 probabilities.
   :rtype: torch.Tensor

   :raises RuntimeError: If channels, spatial size or dtype/device is incompatible.

.. py:class:: ConvNormAct(nn.Module)

   Extract features with convolution, batch normalisation and ReLU.

.. py:method:: ConvNormAct.__init__(self, ic, oc, k=3, d=1)

   Build a padded feature convolution.

   :param ic: int; positive input channels.
   :param oc: int; positive output channels.
   :param k: int; positive kernel side, normally odd.
   :param d: int; positive dilation rate.

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

.. py:method:: ResidualDecoderBlock.__init__(self, ic, sc, oc)

   Construct convolutions and a channel-matching shortcut.

   :param ic: int; positive decoder channels.
   :param sc: int; skip channels, or 0 when no skip will be provided.
   :param oc: int; positive output channels.

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

.. py:method:: AttentionGate.__init__(self, gc, sc, ic)

   Construct context projections and scalar spatial attention.

   :param gc: int; positive context channels.
   :param sc: int; positive encoder channels.
   :param ic: int; positive intermediate channels.

   :returns: None.

.. py:method:: AttentionGate.forward(self, g, s)

   Apply sigmoid attention while retaining the skip shape.

   :param g: torch.Tensor; [B, gate_channels, Hg, Wg] context.
   :param s: torch.Tensor; [B, skip_channels, Hs, Ws] encoder features.

   :returns:

             Weighted skip [B, skip_channels, Hs, Ws], not a
                 one-channel prediction.
   :rtype: torch.Tensor

   :raises RuntimeError: If channels, batch dimensions or devices are incompatible.

.. py:class:: ASPP(nn.Module)

   Combine parallel convolutions at dilation rates 1, 2, 4 and 6.

   The rate-1 branch uses a 1x1 kernel. Projection fuses concatenated features; dropout
   is active only in training mode.

.. py:method:: ASPP.__init__(self, ic, oc)

   Construct four branches and a context projection.

   :param ic: int; positive input channels.
   :param oc: int; output channels, at least 4; branch width is
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

.. py:class:: ImprovedUNet(nn.Module)

   Predict branch logits with a four-channel ResNet-34 U-Net.

   ASPP supplies multiscale context, attention weights skip features, and residual
   blocks fuse them. Layers match training checkpoint keys; construction uses
   weights=None.

.. py:method:: ImprovedUNet.__init__(self)

   Construct encoder and decoder layers.

   :returns: None.

.. py:method:: ImprovedUNet.forward(self, x)

   Decode RGB-D features into full-resolution branch logits.

   :param x: torch.Tensor; floating [B, 4, H, W] batch using training
             preprocessing.

   :returns:

             Raw [B, 1, H, W] logits; sigmoid converts to
                 probabilities.
   :rtype: torch.Tensor

   :raises RuntimeError: If channels, spatial size or dtype/device is incompatible.

.. py:function:: load_checkpoint(model, path, device)

   Load matching parameters strictly and select evaluation mode.

   Loads on CPU with weights_only=False before transfer. Checkpoints must be trusted
   because pickle deserialisation can execute payloads.

   :param model: torch.nn.Module with matching state keys.
   :param path: str or pathlib.Path to a trusted local PyTorch checkpoint.
   :param device: torch.device for the returned model.

   :returns:

             Device-mapped eval model and original
                 checkpoint; accepts model_state_dict or a bare state dictionary.
   :rtype: tuple[torch.nn.Module, dict]

   :raises OSError: If reading fails.
   :raises RuntimeError: If strict loading or device transfer fails.

.. py:function:: save_outputs(rgbim, depthim, mask, outdir, stem)

   Save a prediction, orange overlay and four-panel comparison.

   Suffixes are _pred_mask.png, _overlay.png and _comparison.png. No GT or quantitative
   segmentation accuracy is produced.

   :param rgbim: PIL.Image.Image; preprocessed RGB.
   :param depthim: PIL.Image.Image; same-size L-mode depth.
   :param mask: PIL.Image.Image; same-size L-mode binary prediction, 0/255.
   :param outdir: pathlib.Path; created if needed.
   :param stem: str; filesystem-safe basename without extension.

   :returns: None.

   :raises OSError: If directory creation or PNG writing fails.
   :raises ValueError: If arrays have incompatible dimensions.

.. py:function:: sync(device)

   Finish CUDA kernels before wall-clock timing.

   :param device: torch.device; CPU does nothing, CUDA synchronises that device.

   :returns: None.

   :raises RuntimeError: If CUDA synchronisation fails.

.. py:function:: benchmark(model, samples, name, outroot, device, size, threshold, warmup)

   Measure model-only latency and export winter predictions.

   Caller sets eval mode; inference_mode disables gradients. Timing excludes
   preprocessing, transfer, sigmoid, thresholding and saving. Sample standard deviation
   uses ddof=1 for at least two images; absent means are None.

   :param model: torch.nn.Module; eval model on device mapping RGB-D to branch logits.
   :param samples: list[dict] from discover.
   :param name: str; model label and output subdirectory.
   :param outroot: pathlib.Path; root output directory.
   :param device: torch.device matching model parameters.
   :param size: int; positive input side matching training geometry.
   :param threshold: float in [0, 1]; inclusive sigmoid cutoff.
   :param warmup: int; nonnegative untimed first-sample forward passes.

   :returns:

             Per-image timings and summary in milliseconds;
                 throughput is model-only images/second.
   :rtype: tuple[list[dict], dict]

   :raises OSError: If input reading or prediction saving fails.
   :raises RuntimeError: If inference or synchronisation fails.

.. py:function:: main()

   Load three checkpoints and benchmark paired winter images.

   Requires --data-root and --mlp-checkpoint, --unet-checkpoint, --improved-checkpoint.
   Saves per_image_timings.csv, summary.csv, benchmark_metadata.json and model-specific
   images. Input size comes from CLI, not checkpoint metadata. Requires no GT.

   :returns: None.

   :raises SystemExit: If required CLI arguments are missing or --help is requested.
   :raises FileNotFoundError: If a required path is absent.
   :raises RuntimeError: If pairing, loading or inference fails.
   :raises OSError: If inputs/outputs cannot be accessed.

