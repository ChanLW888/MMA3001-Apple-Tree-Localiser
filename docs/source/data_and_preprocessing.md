# Data and preprocessing

## Pairing

Branch trainers recursively locate lowercase-extension `*.png` files with summer in the stem. `_RGB-`, `_D-` and `_GT-` are replaced by a common token to pair each tree/season/time identifier. Duplicate keys raise errors; incomplete triplets are omitted. Filename agreement alone does not verify spatial registration.

The organised layout is `data/Organised/<Summer|Winter>/<RGB|D|GT>/`. Winter inference pairs winter RGB/depth without GT. There is no independently audited dataset count in this documentation update; earlier exact counts and image dimensions have been removed as unverified historical observations. See the maintained data README for COCO preparation and apple inputs.

## Tensor contract

1. Convert RGB to RGB mode and depth to L-mode grayscale.
2. Centre-crop each to its largest square. A 640x480 frame would yield a central 480x480 crop; this is a geometric example, not a verified inventory.
3. Resize to input_size (default 256): bilinear RGB, nearest-neighbour depth.
4. Divide RGB by 255, then use means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The MLP uses the same scaling as the U-Nets.
5. Divide grayscale depth by 255 and concatenate after RGB: float32 `[4, S, S]` per sample or `[B, 4, S, S]` per batch.
6. Convert GT to L mode without cropping/resizing; require dimensions `(S, S)`. Set target=1 where GT >mask_threshold (127 by default), otherwise 0. Target is float32 `[1, S, S]`.

ImageNet normalisation aligns RGB scaling with the pretrained encoder; depth remains a separate intensity feature. Nearest-neighbour depth interpolation avoids creating intermediate encoded values. L conversion is not metric depth calibration and can discard precision from richer source depth encodings.

GT must already describe the same centred field of view, not merely share dimensions. This assumption is not proven by script checks. A mismatch in GT size raises RuntimeError; a wrong field of view can pass size validation. Model inference must preserve training channel order, geometry and scaling.

## Apple mask distinctions

Traditional apple detection retains original RGB/depth grids and requires equal spatial dimensions. Depth >0 means valid; GT is full-frame nearest-neighbour resized, with foreground >0. RF-DETR does not use depth; its GT foreground cutoff is >127, default alignment centres a square mask in the original frame, and its predicted mode reads exported masks. The predicted comparison driver instead runs the improved U-Net or reuses square masks and projects them centrally; its optional GT control also thresholds >127. Both dilate branch regions using a diameter-100 ellipse by default; differing alignment/cutoffs still create different regions.

Branch GT is not apple-instance annotation. COCO files may contain apple annotations, but authorship, frame correspondence and train/evaluation separation require verification. RF-DETR detections used for comparison are pseudo-ground truth and do not establish real accuracy.
