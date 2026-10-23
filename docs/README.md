# HTML source documentation

Begin with the [reviewer route](source/reviewer_start.md) for a concise guide to
data-free verification, full reproduction prerequisites and claim boundaries.

Sphinx with MyST and Napoleon formats the existing project guides and statically extracted API documentation for all 12 maintained Python modules in `src/`. The AST extractor includes classes, functions, methods and nested helpers, preserving existing docstring parameter, return and exception descriptions. Missing docstrings are explicitly labelled. It never imports project source modules or evaluates their defaults, decorators or annotations.

## Build and view

Run from the repository root with `mma3001` after installing [requirements.txt](../requirements.txt) as described in the [root setup guide](../README.md#environment-and-hardware):

```powershell
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html docs/source docs/_build/html
conda run --no-capture-output -n mma3001 python -m sphinx -E -a -W --keep-going -c docs/static_config -b html -d docs/_build/submission-doctrees docs/source docs/html
```

Open `docs/_build/html/index.html` directly in a browser. The same site is provided for submission at [docs/html/index.html](html/index.html). Submit the whole `docs/html/` directory, including assets and linked pages. `.gitignore` excludes `docs/_build/`; it does not exclude `docs/html/`. No commit or push is performed automatically.


## Configuration and coverage

The new [build configuration](static_config/conf.py) reuses the base `source/conf.py`, then adds static API generation and project metadata. Always use `-c docs/static_config`; the original configuration alone does not regenerate APIs. Every build regenerates `source/api/*.rst` from the current `src/` inventory. Do not edit generated API pages manually.

The landing page links preserved branch guides, the apple workflow/environment guide and the API index. Both `mma3001` and `rfdetr` are explained in [workflows](source/workflows.md). Runtime compatibility, dynamic/inherited APIs and correctness of existing docstrings are not verified by static extraction. No training, inference or preprocessing runs during a documentation build.

The root MIT licence applies only to original author-owned project code, excluding university/Roboflow datasets, third-party weights, RF-DETR code/weights and other external materials. Their own terms apply.

See [verification for this task](html_verification.md). See also the [software test report](../test/test_report.txt) and [test scope](../test/README.md).

The maintained API inventory includes `winter_inference` and `full_grid_search_apple`.
The absent former optimiser is excluded. The grid API documents every helper,
selection direction, fixed branch settings, caching, resume fingerprints and the
known stale held-out result issue. Unit checks do not validate model accuracy.


The new [predicted branch guide](source/predicted_branches.md) and its API page are
part of the site. [Final audit](final_audit.md) records rubric evidence and remaining
submission requirements; [AI disclosure](ai_disclosure.md) identifies outstanding
student reflection. Historical verification claims are superseded by this audit's
actual rerun evidence.
