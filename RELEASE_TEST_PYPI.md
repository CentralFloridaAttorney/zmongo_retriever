
---

### 📦 `RELEASE_TESTPYPI.md`

# 🚀 How to Publish a New Release to TestPyPI

This guide walks you through building and uploading a fresh release of `zmongo-retriever` (and its modules like `zmongo_toolbag`, `zai_toolbag`) to **TestPyPI**.

---

## ✅ 1. Update Your Version

Make sure you **bump the version** in your `pyproject.toml`:

```toml
[project]
name = "zmongo-retriever"
version = "0.1.4"  # 🔁 Update this
```

---

## ✅ 2. Clean Up Previous Builds

```bash
rm -rf dist/ build/ *.egg-info
```

---

## ✅ 3. Build the Package

```bash
python3 -m build
```

---

## ✅ 4. Upload to TestPyPI

```bash
python3 -m twine upload --repository testpypi dist/*
```

> You’ll need your TestPyPI credentials (`~/.pypirc` should be configured).
> Example `.pypirc`:
```ini
[testpypi]
  username = __token__
  password = pypi-<your-token-here>
```

---

## ✅ 5. Test the Installed Package in a New Virtual Env

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple zmongo-retriever
```

> 💡 If you need dev dependencies:
```bash
pip install "zmongo-retriever[dev]" --index-url https://test.pypi.org/simple/
```

---

## ✅ 6. Verify Installation

```bash
python -c "from zmongo_toolbag.zmongo import ZMongo; print(ZMongo)"
```

Or run any test script or REPL import to confirm the install is working as expected.

---

## 🧪 Optional: Run Tests with the Installed Package

```bash
pytest tests/
```

---

## 🛠 Notes

- Ensure `pyproject.toml` includes both `zmongo_toolbag/` and `zai_toolbag/` in the package list.
- Confirm all required files (e.g., `.env.example`, README.md) are included via `MANIFEST.in`.
- Use semantic versioning: `MAJOR.MINOR.PATCH`.

---