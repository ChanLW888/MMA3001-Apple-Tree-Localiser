"""Configure static Sphinx branch-segmentation documentation.

Uses MyST Markdown and Alabaster. No project model modules are imported; building this
guide does not train or run detectors.

Example:
    python -m sphinx -b html docs/source docs/_build/html
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parents[1]

project = "Apple Tree Localiser — Branch Segmentation"
author = "Project contributors"
copyright = "2026, Project contributors"

# Static Markdown pages avoid importing scripts with training or filesystem side effects.
extensions = ["myst_parser"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "alabaster"
html_title = "Branch Segmentation Documentation"
html_static_path = []
