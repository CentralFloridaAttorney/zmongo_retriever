Perfect — here’s a **one-page `pip install` cheat sheet** with the options you’ll actually care about for development 👇

---

# 🐍 `pip install` Cheat Sheet

## 📦 Installing Your Own Package

```bash
pip install .           # Normal install (copies code into site-packages)
pip install -e .        # Editable install (links code, reflects changes live)
```

---

## 📂 Installing from Files

```bash
pip install -r requirements.txt      # From requirements file
pip install -r requirements-dev.txt  # Dev/test/lint dependencies
```

---

## ➕ Extras (optional groups of deps)

In `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "mypy"]
docs = ["sphinx"]
```

Install them:

```bash
pip install .[dev]        # Install dev tools
pip install .[dev,docs]   # Multiple extras
```

---

## 🔄 Upgrading / Reinstalling

```bash
pip install -U requests        # Upgrade to latest
pip install --force-reinstall .  # Reinstall even if already installed
```

---

## 🚫 Controlling Dependencies

```bash
pip install --no-deps package   # Skip installing dependencies
pip install --upgrade-strategy eager   # Upgrade all deps
```

---

## 📍 Sources & Indexes

```bash
pip install package==1.2.3  # Specific version
pip install git+https://github.com/user/repo.git
pip install --find-links ./wheels mypkg   # Local wheelhouse
pip install --extra-index-url https://test.pypi.org/simple mypkg
```

---

## 📂 Custom Install Location

```bash
pip install -t ./vendor requests   # Install into folder (e.g. AWS Lambda)
```

---

✅ **Everyday workflow for dev projects:**

1. `pip install -e .[dev]` → editable install w/ dev deps
2. `pip install -r requirements.txt` → sync pinned deps
3. `pip install -U package` → upgrade when needed

---

Would you like me to make you a **ready-to-paste `requirements-dev.txt` and `pyproject.toml` block** so you can just run `pip install -e .[dev]` in your current project setup?
