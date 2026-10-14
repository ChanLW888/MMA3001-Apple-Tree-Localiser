"""Sphinx configuration for branch-segmentation documentation."""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parents[1]

project = "Apple Tree Localiser — Branch Segmentation"
author = "Project contributors"
copyright = "2026, Project contributors"

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
