from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import csv
import re


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "Organised"

SEASON = "Summer"

RGB_DIR = DATASET / SEASON / "RGB"
DEPTH_DIR = DATASET / SEASON / "D"
GT_DIR = DATASET / SEASON / "GT"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "apple_detection"
ANNOTATED_DIR = OUTPUT_ROOT / "annotated"
CSV_PATH = OUTPUT_ROOT / "apple_counts.csv"


# Apple detection parameters
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


def parse_rgb_filename(path):
    """
    Match filenames such as:
    R10N12_Summer_RGB-13-36-51.png
    """

    pattern = (
        r"^(?P<tree>R\d+N\d+)_"
        r"(?P<season>Summer|Winter)_"
        r"RGB-(?P<time>.+)\.png$"
    )

    match = re.match(pattern, path.name)

    if not match:
        return None

    return {
        "tree": match.group("tree"),
        "season": match.group("season"),
        "time": match.group("time"),
    }


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_depth(path):
    depth = np.array(Image.open(path))

    # Some depth files are stored with three channels.
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def load_gt(path, target_shape):
    gt = np.array(Image.open(path).convert("L"))

    # Resize GT to match RGB/depth resolution if needed.
    if gt.shape != target_shape:
        gt = cv2.resize(
            gt,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return gt


def detect_red_mask(rgb, depth):
    """
    Detect red pixels.

    Depth is only used as a validity mask:
    pixels with depth <= 0 are ignored.
    """

    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)

    mask = (
        (R > RED_THRESHOLD)
        & (R > G * RED_DOMINANCE)
        & (R > B * RED_DOMINANCE)
        & (depth > 0)
    )

    return mask.astype(np.uint8)


def clean_red_mask(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def create_branch_zone(gt):
    branch_mask = (gt > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BRANCH_DILATION, BRANCH_DILATION),
    )

    return cv2.dilate(branch_mask, kernel)


def find_apple_candidates(rgb, depth, gt):
    """
    Find candidate apples using connected red regions.

    The detection uses:
    - RGB redness
    - valid depth
    - area
    - aspect ratio
    - circularity
    - overlap with the GT branch zone
    """

    red_mask = detect_red_mask(rgb, depth)
    red_mask = clean_red_mask(red_mask)

    branch_zone = create_branch_zone(gt)

    # Get connected components once.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask,
        connectivity=8,
    )

    candidates = []

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        # Quick area rejection before doing any expensive processing.
        if area < MIN_APPLE_AREA or area > MAX_APPLE_AREA:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Reject regions that are too elongated to be an apple.
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # Work only inside the component's bounding box instead of
        # creating a full-image mask for every component.
        component = (
            labels[y:y + h, x:x + w] == label
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        # Only compare the branch mask within the component bbox.
        local_branch = branch_zone[
            y:y + h,
            x:x + w
        ]

        overlap_pixels = np.count_nonzero(
            (component > 0) & (local_branch > 0)
        )

        branch_overlap = overlap_pixels / max(area, 1)

        if branch_overlap < MIN_BRANCH_OVERLAP:
            continue

        # Only calculate the red score for pixels belonging
        # to this component.
        local_rgb = rgb[
            y:y + h,
            x:x + w
        ]

        component_pixels = component > 0

        region = local_rgb[component_pixels].astype(
            np.float32
        )

        R = region[:, 0]
        G = region[:, 1]
        B = region[:, 2]

        red_score = np.mean(
            R / (G + B + 1)
        )

        candidates.append({
            "label": label,
            "area": int(area),
            "bbox": (
                int(x),
                int(y),
                int(w),
                int(h),
            ),
            "centroid": tuple(
                centroids[label]
            ),
            "circularity": float(
                circularity
            ),
            "aspect_ratio": float(
                aspect_ratio
            ),
            "branch_overlap": float(
                branch_overlap
            ),
            "red_score": float(
                red_score
            ),
        })

    return candidates


def draw_annotations(rgb, candidates):
    """
    Save the RGB image with detected apples marked.
    """

    annotated = rgb.copy()

    for i, apple in enumerate(
        candidates,
        start=1,
    ):

        x, y, w, h = apple["bbox"]
        cx, cy = apple["centroid"]

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

        cv2.circle(
            annotated,
            (int(cx), int(cy)),
            3,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            annotated,
            str(i),
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Apples: {len(candidates)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def get_matching_files(rgb_path, info):

    depth_path = (
        DEPTH_DIR
        / f'{info["tree"]}_{info["season"]}_D-{info["time"]}.png'
    )

    gt_path = (
        GT_DIR
        / f'{info["tree"]}_{info["season"]}_GT-{info["time"]}.png'
    )

    if not depth_path.exists():
        return None, None

    if not gt_path.exists():
        return None, None

    return depth_path, gt_path


def process_frame(rgb_path, info):

    depth_path, gt_path = get_matching_files(
        rgb_path,
        info,
    )

    if depth_path is None:
        return None

    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)
    gt = load_gt(
        gt_path,
        depth.shape,
    )

    candidates = find_apple_candidates(
        rgb,
        depth,
        gt,
    )

    annotated = draw_annotations(
        rgb,
        candidates,
    )

    # Save annotated image.
    output_dir = (
        ANNOTATED_DIR / info["tree"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / rgb_path.name
    )

    Image.fromarray(
        annotated
    ).save(output_path)

    return {
        "tree": info["tree"],
        "season": info["season"],
        "time": info["time"],
        "apple_count": len(candidates),
        "annotated_path": output_path,
    }


def find_all_frames():

    frames = []

    for rgb_path in RGB_DIR.glob("*.png"):

        info = parse_rgb_filename(
            rgb_path
        )

        if info is None:
            continue

        if info["season"] != SEASON:
            continue

        depth_path, gt_path = get_matching_files(
            rgb_path,
            info,
        )

        if depth_path is None:
            continue

        if gt_path is None:
            continue

        frames.append(
            (rgb_path, info)
        )

    frames.sort(
        key=lambda item: (
            item[1]["tree"],
            item[1]["time"],
        )
    )

    return frames


def save_csv(results):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Tree",
            "Season",
            "Time",
            "Apple_Count",
            "Annotated_Image",
        ])

        for result in results:

            writer.writerow([
                result["tree"],
                result["season"],
                result["time"],
                result["apple_count"],
                result["annotated_path"],
            ])


def main():

    frames = find_all_frames()

    if not frames:

        print("No matching frames found.")
        print(f"Dataset: {DATASET}")

        return

    print(
        f"\nProcessing {len(frames)} frames..."
    )

    results = []
    current_tree = None

    for rgb_path, info in frames:

        if info["tree"] != current_tree:

            current_tree = info["tree"]

            print(f"\n{current_tree}")

        result = process_frame(
            rgb_path,
            info,
        )

        if result is None:
            continue

        results.append(result)

        count = result["apple_count"]
        label = (
            "apple"
            if count == 1
            else "apples"
        )

        print(
            f"  {info['time']} → "
            f"{count} {label}"
        )

    save_csv(results)

    if results:

        counts = [
            result["apple_count"]
            for result in results
        ]

        print("\nDone.")
        print(
            f"Frames processed: "
            f"{len(results)}"
        )
        print(
            f"Average apples/frame: "
            f"{np.mean(counts):.2f}"
        )
        print(
            f"Results: {CSV_PATH}"
        )


if __name__ == "__main__":
    main()