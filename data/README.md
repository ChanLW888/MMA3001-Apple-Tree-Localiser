# Data

Place the following two dataset folders inside this directory:

```text
data/
├── Dataset/
└── Dataset (With Summer GT)/
```

The organiser script uses `Dataset (With Summer GT)` as its input.

## Organise the Dataset

From the **project root**, run:

```bash
python src/dataset_organiser.py
```

This will generate:

```text
data/
├── Dataset/
├── Dataset (With Summer GT)/
└── Organised/
    ├── Winter/
    │   ├── RGB/
    │   ├── D/
    │   └── GT/
    └── Summer/
        ├── RGB/
        ├── D/
        └── GT/
```

The script identifies files based on the naming format:

```text
<TreeID>_<Season>_<Type>-<Time>.png
```

For example:

```text
R01N01_Summer_RGB-12-40-50.png
```

Supported seasons:

- `Winter`
- `Summer`

Supported image types:

- `RGB`
- `D`
- `GT`

The script also performs a basic check on one image from each category and reports the number of processed and skipped files.

> **Note:** The dataset files are not included in the repository due to their size. Obtain the datasets separately and place them in the `data/` directory before running the script.
