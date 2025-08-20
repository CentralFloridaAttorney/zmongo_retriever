Alright 👍 — here’s a **ready-to-paste setup** so your project can support

```bash
pip install -e .[dev]
```

---

## 1️⃣ `pyproject.toml` block

Add this inside your `pyproject.toml` (under the main `[project]` section):

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "black",
    "isort",
    "mypy",
    "flake8",
    "pre-commit"
]
```

This defines a **`dev` extra group**.

---

## 2️⃣ `requirements-dev.txt`

Also create a `requirements-dev.txt` file (in your project root) with pinned versions for reproducibility:

```txt
# Testing
pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0

# Code style & linting
black==24.4.2
isort==5.13.2
flake8==7.1.0
mypy==1.10.0

# Git hooks
pre-commit==3.7.1
```

Then you can install them in two ways:

* **Editable install with extras (preferred for dev):**

  ```bash
  pip install -e .[dev]
  ```
  
  ```bash
  pip install -e .[docs]
  ```
* **From pinned file (CI/CD / reproducible builds):**

  ```bash
  pip install -r requirements-dev.txt
  ```

---

✅ With this setup:

* Day-to-day dev → `pip install -e .[dev]`
* CI builds → `pip install -r requirements-dev.txt`
* Both stay in sync, since the `requirements-dev.txt` pins versions and the `pyproject.toml` declares groups.

---

Do you want me to also set up a **`pre-commit` config** so you automatically run black/flake8/mypy before every commit? That way your repo always stays clean without manual checks.
