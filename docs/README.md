# HTML source documentation

Sphinx with MyST and Napoleon formats the existing project guides and statically extracted API documentation for all 11 maintained Python modules in `src/`. The AST extractor includes classes, functions, methods and nested helpers, preserving existing docstring parameter, return and exception descriptions. Missing docstrings are explicitly labelled. It never imports project source modules or evaluates their defaults, decorators or annotations.

## Build and view

Run from the repository root with the existing `mma3001` environment, which already contains Sphinx/MyST:

```powershell
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
```

Open `docs/_build/html/index.html` directly in a browser. The same site is provided for submission at [docs/html/index.html](html/index.html). Submit the whole `docs/html/` directory, including assets and linked pages. `.gitignore` excludes `docs/_build/`; it does not exclude `docs/html/`. No commit or push is performed automatically.

The existing requirements are listed in `docs/requirements-docs.txt`; no packages were installed or upgraded for this task. If dependencies are unavailable, the commands report the missing package; neither application environment YAML needs editing.

## Configuration and coverage

The new [build configuration](static_config/conf.py) reuses the unchanged original `source/conf.py`, then adds static API generation and project metadata. Always use `-c docs/static_config`; the original configuration alone does not regenerate APIs. Every build regenerates `source/api/*.rst` from the current `src/` inventory. Do not edit generated API pages manually.

The landing page links preserved branch guides, the apple workflow/environment guide and the API index. Both `mma3001` and `rfdetr` are explained in [workflows](source/workflows.md). Runtime compatibility, dynamic/inherited APIs and correctness of existing docstrings are not verified by static extraction. No training, inference or preprocessing runs during a documentation build.

The root MIT licence applies only to original author-owned project code, excluding university/Roboflow datasets, third-party weights, RF-DETR code/weights and other external materials. Their own terms apply.

See [verification for this task](html_verification.md). The older `documentation_verification.md` describes a previous task and its historical checks, not the changes authorised or performed here.
