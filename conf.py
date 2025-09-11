# docs/source/conf.py
from __future__ import annotations
import os
import sys
from datetime import datetime

# If using src/ layout:
sys.path.insert(0, os.path.abspath("../../src"))

project = "YourProjectName"
author = "Your Name"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# Allow Markdown
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# Autodoc / Autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # move type hints into the description
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx to cross-link big projects' docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "pydantic": ("https://docs.pydantic.dev/latest/", {}),
}

# Theme
html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]