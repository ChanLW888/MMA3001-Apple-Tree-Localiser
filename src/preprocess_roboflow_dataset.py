from pathlib import Path
import json
import shutil


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "data"
    / "apple tree.v1i.coco-segmentation"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "apple_tree_clean"
)

# Original dataset splits
TRAIN_SOURCE = SOURCE_DIR / "train"
VALID_SOURCE = SOURCE_DIR / "export"

TRAIN_JSON = TRAIN_SOURCE / "_annotations.coco.json"
VALID_JSON = VALID_SOURCE / "_annotations.coco.json"


# ============================================================
# CHECK INPUT DATA
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_DIR}"
    )

if not TRAIN_SOURCE.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_SOURCE}"
    )

if not VALID_SOURCE.exists():
    raise FileNotFoundError(
        f"Validation source folder not found:\n{VALID_SOURCE}"
    )

if not TRAIN_JSON.exists():
    raise FileNotFoundError(
        f"Training annotations not found:\n{TRAIN_JSON}"
    )

if not VALID_JSON.exists():
    raise FileNotFoundError(
        f"Validation annotations not found:\n{VALID_JSON}"
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(source_dir, annotation_file, output_dir):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {source_dir.name}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load COCO annotations
    # --------------------------------------------------------

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    print(f"Images in original COCO: {len(coco.get('images', []))}")
    print(f"Annotations in original COCO: {len(coco.get('annotations', []))}")
    print(f"Categories in original COCO: {len(coco.get('categories', []))}")

    # --------------------------------------------------------
    # Find apple categories
    # --------------------------------------------------------

    apple_categories = []

    for category in coco.get("categories", []):

        name = category.get("name", "").strip().lower()

        if name.startswith("apple-"):
            apple_categories.append(category)

    print(f"\nApple categories found: {len(apple_categories)}")

    if not apple_categories:
        raise RuntimeError(
            "No categories beginning with 'apple-' were found."
        )

    apple_category_ids = {
        category["id"]
        for category in apple_categories
    }

    # --------------------------------------------------------
    # Print category names
    # --------------------------------------------------------

    print("\nApple categories:")

    for category in apple_categories:
        print(
            f"  {category['id']:>4} -> {category['name']}"
        )

    # --------------------------------------------------------
    # Create new COCO structure
    # --------------------------------------------------------

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "apple",
                "supercategory": "apple"
            }
        ]
    }

    # --------------------------------------------------------
    # Keep all images
    #
    # This is important:
    # images containing no apples remain in the dataset.
    # They act as negative/background examples.
    # --------------------------------------------------------

    for image in coco.get("images", []):
        new_coco["images"].append(image)

    # --------------------------------------------------------
    # Keep ONLY apple annotations
    #
    # Each original apple annotation remains a separate
    # instance.
    # --------------------------------------------------------

    new_annotation_id = 1

    kept_annotations = 0
    removed_annotations = 0

    for annotation in coco.get("annotations", []):

        original_category_id = annotation.get("category_id")

        if original_category_id in apple_category_ids:

            new_annotation = annotation.copy()

            # Convert every apple-* category to class 1 = apple
            new_annotation["category_id"] = 1

            # Give annotations clean sequential IDs
            new_annotation["id"] = new_annotation_id

            new_coco["annotations"].append(
                new_annotation
            )

            new_annotation_id += 1
            kept_annotations += 1

        else:
            removed_annotations += 1

    print("\nAnnotation processing:")
    print(f"  Apple instances kept   : {kept_annotations}")
    print(f"  Non-apple annotations removed: {removed_annotations}")

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    copied_images = 0
    missing_images = 0

    print("\nCopying images...")

    for image in new_coco["images"]:

        file_name = image["file_name"]

        source_image = source_dir / file_name
        output_image = output_dir / file_name

        if not source_image.exists():

            print(
                f"WARNING: image not found: {source_image}"
            )

            missing_images += 1
            continue

        output_image.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source_image,
            output_image
        )

        copied_images += 1

    # --------------------------------------------------------
    # Save new COCO JSON
    # --------------------------------------------------------

    output_json = output_dir / "_annotations.coco.json"

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            new_coco,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nSplit complete:")
    print(f"  Images in COCO       : {len(new_coco['images'])}")
    print(f"  Images copied        : {copied_images}")
    print(f"  Missing images       : {missing_images}")
    print(f"  Apple instances      : {kept_annotations}")
    print(f"  Output annotations   : {output_json}")

    return {
        "images": len(new_coco["images"]),
        "apple_instances": kept_annotations,
        "copied_images": copied_images,
        "missing_images": missing_images,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("APPLE DATASET PREPROCESSING")
    print("=" * 60)

    print(f"\nOriginal dataset:")
    print(SOURCE_DIR)

    print(f"\nClean dataset:")
    print(OUTPUT_DIR)

    print("\nThe original dataset will NOT be modified.")

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    train_output = OUTPUT_DIR / "train"
    valid_output = OUTPUT_DIR / "valid"

    # --------------------------------------------------------
    # Process train
    # --------------------------------------------------------

    train_stats = process_split(
        source_dir=TRAIN_SOURCE,
        annotation_file=TRAIN_JSON,
        output_dir=train_output
    )

    # --------------------------------------------------------
    # Process validation
    # --------------------------------------------------------

    valid_stats = process_split(
        source_dir=VALID_SOURCE,
        annotation_file=VALID_JSON,
        output_dir=valid_output
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print("\nTRAIN:")
    print(f"  Images          : {train_stats['images']}")
    print(f"  Apple instances : {train_stats['apple_instances']}")

    print("\nVALID:")
    print(f"  Images          : {valid_stats['images']}")
    print(f"  Apple instances : {valid_stats['apple_instances']}")

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nClasses:")
    print("  1 -> apple")

    print("=" * 60)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()