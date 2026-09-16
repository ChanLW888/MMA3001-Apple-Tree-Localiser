# Apple Localisation for MMA3001 Project

A computer vision pipeline for detecting and localising apples in the TreeID dataset using RGB images, depth maps, and tree/branch ground-truth information.

This project is part of the **MMA3001 project** and focuses on identifying apple candidates in orchard images and producing annotated images and apple-count results.

## Project Overview

The goal of this project is to automatically identify apples from images of apple trees.

The current detection pipeline combines:

* **RGB images** to identify red apple-like regions
* **Depth maps** to remove pixels without valid depth
* **Ground-truth tree/branch masks** to restrict detections to regions close to the tree structure
* **Shape filtering** to reject regions that are unlikely to be apples

The current method is primarily a **2D image-based detection approach**. Although depth is loaded, the actual depth values are currently only used as a validity mask (`depth > 0`) rather than being used to estimate the 3D position of each apple.

## Detection Pipeline

The detection process is:

```text
RGB Image
    │
    ├── Red colour detection
    │
    ├── Depth validity check
    │
    ▼
Red Mask
    │
    ├── Morphological filtering
    │
    ▼
Connected Components
    │
    ├── Area filtering
    ├── Aspect ratio filtering
    ├── Circularity filtering
    ├── Branch-region overlap
    └── Red colour score
    │
    ▼
Apple Candidates
    │
    ├── Bounding boxes
    ├── Centre points
    └── Apple count
    │
    ▼
Annotated RGB Images + CSV
```

## Apple Detection

Red regions are identified using the RGB channels.

A pixel is considered part of the initial red mask when:

```text
R > RED_THRESHOLD
R > G × RED_DOMINANCE
R > B × RED_DOMINANCE
depth > 0
```

The resulting mask is cleaned using morphological opening and closing before connected components are extracted.

Each connected component is then filtered using:

| Filter         | Purpose                                              |
| -------------- | ---------------------------------------------------- |
| Area           | Removes regions that are too small or too large      |
| Aspect ratio   | Rejects elongated regions                            |
| Circularity    | Favors roughly circular apple-shaped regions         |
| Branch overlap | Removes red regions far from the tree                |
| Red score      | Measures how strongly the region is dominated by red |

## Dataset Structure

The dataset is expected to be organised as:

```text
data/
└── Organised/
    └── Summer/
        ├── RGB/
        │   ├── R10N12_Summer_RGB-13-36-51.png
        │   └── ...
        │
        ├── D/
        │   ├── R10N12_Summer_D-13-36-51.png
        │   └── ...
        │
        └── GT/
            ├── R10N12_Summer_GT-13-36-51.png
            └── ...
```

The RGB, depth and GT files are matched using:

* Tree ID
* Season
* Timestamp

For example:

```text
R10N12_Summer_RGB-13-36-51.png
R10N12_Summer_D-13-36-51.png
R10N12_Summer_GT-13-36-51.png
```

represent the same frame.

## Repository Structure

```text
Git_repo_location/
│
├── data/
│   └── Organised/
│       └── ...
│
├── src/
│   └── apple_predictor_v1.py
│
├── outputs/
│   └── apple_detection/
│       ├── annotated/
│       └── apple_counts.csv
│
├── .gitignore
└── README.md
```

> The dataset and generated outputs are excluded from Git using `.gitignore` because of their size.

## Requirements

The project uses Python and the following packages:

* Python
* NumPy
* OpenCV
* Pillow

Install the dependencies with:

```bash
pip install numpy opencv-python pillow
```

If using the existing Conda environment:

```bash
conda activate tree3d
```

## Running the Detector

From the repository root:

```bash
python src/apple_predictor_v1.py
```

The script will search the configured dataset directory for matching RGB, depth and GT images.

The season can be changed in:

```python
SEASON = "Summer"
```

For example:

```python
SEASON = "Winter"
```

## Output

Annotated images are saved under:

```text
outputs/apple_detection/annotated/
```

Each detected apple is marked with:

* A bounding box
* A centre point
* A detection number

The image also displays the total number of detected apples.

Example:

```text
Apples: 4
```

A CSV file containing the detection results is saved to:

```text
outputs/apple_detection/apple_counts.csv
```

The CSV contains:

```text
Tree
Season
Time
Apple_Count
Annotated_Image
```

## Console Output

The detector provides a concise summary while processing:

```text
Processing 562 frames...

R10N12
  13-36-51 → 3 apples
  13-37-02 → 4 apples
  13-37-13 → 2 apples

R10N13
  13-38-01 → 1 apple
  13-38-12 → 3 apples

Done.
Frames processed: 562
Average apples/frame: 2.47
Results: outputs/apple_detection/apple_counts.csv
```

## Configuration

The main detection parameters are defined near the top of `apple_predictor_v1.py`.

```python
RED_THRESHOLD = 50
RED_DOMINANCE = 1.5

MIN_RED_AREA = 15
MIN_APPLE_AREA = 30
MAX_APPLE_AREA = 5000

MIN_CIRCULARITY = 0.20
MAX_ASPECT_RATIO = 2.5

BRANCH_DILATION = 100
MIN_BRANCH_OVERLAP = 0.05

MORPH_KERNEL_SIZE = 5
```

These parameters can be adjusted depending on the appearance of the apples and the amount of noise in the images.

## Current Limitations

The current implementation has several limitations.

### Colour dependence

The detector relies heavily on red colour information. Apples that are:

* poorly illuminated
* occluded
* not sufficiently red
* affected by reflections or shadows

may not be detected reliably.

### False positives

Other red objects or regions can potentially be detected as apples. Shape and branch-overlap filtering are used to reduce these false positives, but they cannot eliminate them completely.

### Depth is not currently used for 3D localisation

The depth map is currently used only to determine whether a pixel has valid depth:

```python
depth > 0
```

The actual depth value is not currently used to calculate the apple's 3D position.

Therefore, the current system should be considered a **2D apple detection/localisation pipeline**, rather than a full 3D apple localisation system.

### Ground-truth dependence

The branch-region filtering currently makes use of the GT mask to determine whether a detected red region is sufficiently close to the tree structure.

This means the current approach relies on labelled information during detection and is not yet a completely independent deployment pipeline.

## Future Work

Potential improvements include:

* Using depth values to estimate the 3D position of each detected apple
* Combining apple detections with reconstructed 3D geometry
* Improving detection of partially occluded apples
* Using a learned object-detection or segmentation model
* Reducing dependence on manually labelled GT masks
* Combining RGB detection with LiDAR or reconstructed point clouds
* Evaluating detection performance against labelled apple locations
* Comparing traditional image-processing methods with neural-network-based approaches

## Project Direction

A longer-term goal is to investigate how **2D apple detection can be combined with 3D reconstruction** to obtain the spatial location of apples within an orchard.

The current detector therefore provides the image-space apple candidates that can potentially be associated with depth or reconstructed 3D information in future stages of the project.

## Author

Chan Lik Wai
