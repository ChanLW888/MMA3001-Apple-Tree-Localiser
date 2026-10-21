# Branch segmentation documentation

The Sphinx site provides a static guide to RGB-D branch segmentation. Apple detection, RF-DETR and detector comparison are documented in the [root README](../README.md), [source README](../src/README.md) and [output README](../outputs/README.md). The site does not import model scripts or generate API pages from their docstrings.

## Build HTML

With documentation dependencies already available, run from the repository root:

```powershell
python -m sphinx -b html docs/source docs/_build/html
```

For a separately authorised environment setup, the existing dependency specification supports:

```powershell
python -m pip install -r docs/requirements-docs.txt
```

No installation or build was performed during the documentation-only update. Open `docs/_build/html/index.html` after building; generated HTML is ignored by Git. No hosted documentation URL is configured in the maintained files.

## Maintained pages

- [Overview](source/overview.md)
- [Data and preprocessing](source/data_and_preprocessing.md)
- [Architectures](source/models.md)
- [Training and evaluation](source/training_and_evaluation.md)
- [Usage](source/usage.md)

The Sphinx configuration is [conf.py](source/conf.py). [Documentation verification](documentation_verification.md) records inventory, AST/syntax checks, corrected claims and pre-existing issues. Python modules use Google-style docstrings; the site remains a manually maintained branch guide, so update its examples when filenames or interfaces change.
