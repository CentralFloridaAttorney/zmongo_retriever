Awesome—let’s set you up with a **Sphinx** docs stack like the big Python projects (NumPy, SciPy, Django style): autodoc, autosummary, Napoleon (Google/NumPy docstrings), MyST Markdown, intersphinx, copy‑button, and a clean theme. You’ll be able to build locally and on Read the Docs.

---

# Step 1) Add a `docs` extra (so installs are one-liners)

In your `pyproject.toml` (same section where you added `dev`), add:

```toml
[project.optional-dependencies]
docs = [
  "sphinx~=7.4",
  "sphinx-autobuild~=2024.4.16",
  "myst-parser~=3.0",
  "sphinx-copybutton~=0.5.2",
  "sphinx-autodoc-typehints~=2.5.0",
  "furo~=2024.8.6"
]
```

Install:

```bash
pip install -e .[docs]
```

> `furo` is a modern theme; swap to `sphinx-rtd-theme` if you prefer Read‑the‑Docs look.

---

# Step 2) Scaffold the Sphinx project

From repo root (assuming **src/** layout, e.g. `src/zai_core`):

```bash
sphinx-quickstart docs --sep -q -p "YourProjectName" -a "Your Name" -v "0.1.0" --makefile --no-batchfile
```

* `--sep` → `docs/source` and `docs/build` separation (common in big projects).
* If on Windows, `--no-batchfile` is optional; you can keep `make.bat` if you like.

---

# Step 3) Replace `docs/source/conf.py` with this tuned config

```python
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

# Treat warnings as errors in CI if you want fails on broken refs:
# nitpicky = True
```

---

# Step 4) Create a clean `index.rst` with a top-level ToC

Replace `docs/source/index.rst`:

```rst
Welcome to YourProjectName’s documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
```

Add a simple Markdown quickstart page:

```bash
echo "# Getting Started

Install:

\`\`\`bash
pip install yourproject
\`\`\`

Basic usage:

\`\`\`python
from yourproject import something
\`\`\`
" > docs/source/getting-started.md
```

---

# Step 5) Generate API stubs (autosummary) for your package

Assuming your code lives in `src/yourproject`:

```bash
sphinx-apidoc -o docs/source/api src/yourproject --module-first --force
```

That generates `*.rst` files under `docs/source/api`. Ensure `docs/source/api/index.rst` exists; if not, add:

```rst
API Reference
=============

.. toctree::
   :maxdepth: 2
   :glob:

   yourproject*
```

> Re-run `sphinx-apidoc` whenever public modules change. In bigger projects, teams script this in `Makefile` or `nox`/`tox`.

---

# Step 6) Build locally

```bash
# One-off build
python -m sphinx -b html docs/source docs/build/html

# Live-reload while editing docs
sphinx-autobuild docs/source docs/build/html
```

Open `docs/build/html/index.html`.

---

# Step 7) Read the Docs (like the pros)

Add `.readthedocs.yaml` at repo root:

```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/source/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

Push to GitHub/GitLab, import the repo in Read the Docs, and builds will run with `pip install .[docs]`.

---

# Step 8) Docstring style (what big projects use)

Pick **Google** or **NumPy** style—both supported via `napoleon`.

**Google style example:**

```python
def insert_document(doc: dict) -> str:
    """Insert a document.

    Args:
        doc: The document to insert.

    Returns:
        The inserted document ID.
    """
```

**NumPy style example:**

```python
def insert_document(doc: dict) -> str:
    """Insert a document.

    Parameters
    ----------
    doc : dict
        The document to insert.

    Returns
    -------
    str
        The inserted document ID.
    """
```

---

# (Optional) Nice-to-haves for “enterprise” polish

* **Copy-button** already added; it hides `>>>` prompts automatically.
* **Mermaid diagrams**: `pip install sphinxcontrib-mermaid` and add `"sphinxcontrib.mermaid"` to `extensions`, then use:

  ````md
  ```mermaid
  graph TD
    A[User] --> B[Your API]
  ````

  ```
  ```
* **Doctest** included: put runnable examples in docstrings and run `-b doctest`.
* **Nitpicky CI**: set `nitpicky = True` in `conf.py` and fix missing refs in PRs.

---

## Quick recap (just 3 commands to get rolling)

```bash
pip install -e .[docs]
sphinx-quickstart docs --sep -q -p "zmongo_retriever" -a "John M. Iriye, Esq." -v "0.1.6"
sphinx-apidoc -o docs/source/api src/yourproject
python -m sphinx -b html docs/source docs/build/html
```

Want me to tailor the `sphinx-apidoc` paths and `conf.py` `sys.path.insert` for **your exact repo layout** (e.g., `src/zai_core`, multiple packages like `zmongo_toolbag`, `zai_core`, etc.) and add a tiny build script `scripts/build_docs.py` to regenerate stubs + build in one go?
