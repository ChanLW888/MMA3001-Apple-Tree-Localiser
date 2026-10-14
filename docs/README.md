# Branch Segmentation Documentation

This Sphinx project documents **only the branch-segmentation part** of the
Apple Tree Localiser repository. Apple detection, RF-DETR, and other apple
scripts are deliberately excluded for now.

## Build the HTML website

Run these commands from the repository root:

```bash
python -m pip install -r docs/requirements-docs.txt
python -m sphinx -b html docs/source docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

## Add this to your repository

Copy the contents of this package into your repository's `docs/` directory.
If you already have a `docs/` folder, back it up first and merge the files
rather than overwriting unrelated work.

The API page is intentionally written as a branch-segmentation guide rather
than importing every Python script. This prevents Sphinx from trying to import
apple scripts or optional packages such as `rfdetr` while generating these
pages. The model pages are maintained documentation and do not require the
training environment to execute.

## Script names

The commands in the source guide use the branch-training filenames discussed
so far. If the filenames in your current `src/` folder differ, update the
examples in `source/usage.md` to match your actual filenames.
