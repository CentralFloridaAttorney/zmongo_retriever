Nice — thanks for the dump. A few quick notes from it:

* You already have `zmongo_retriever 0.1.6` installed (looks like a normal install, not editable).
* Your `setuptools` is **65.5.0** (fine), and you’ve got Sphinx/MkDocs bits ready.
* If you want a **wheel saved to `dist/`** (instead of a temp folder), build explicitly.

Here’s exactly what to do in **PowerShell (one line at a time)**:

### 1) Prep the build toolchain

```powershell
python -m pip install -U pip setuptools wheel build twine
```

### 2) Build both sdist + wheel (outputs to `.\dist\`)

```powershell
python -m build
```

### 3) See what you built

```powershell
Get-ChildItem -Path .\dist\
```

You should see files like:

```
dist\zmongo_retriever-0.1.6-py3-none-any.whl
dist\zmongo_retriever-0.1.6.tar.gz
```

### 4) (Optional) Build only a wheel (also goes to `.\dist\`)

```powershell
pip wheel . -w dist
```

### 5) Install the wheel you just built

```powershell
pip install --force-reinstall .\dist\zmongo_retriever-0.1.6-py3-none-any.whl
```

### 6) Switch to editable dev install (if you want live code changes)

```powershell
pip uninstall -y zmongo_retriever
```

```powershell
pip install -e .[dev]
```

### 7) Verify your package is importable

```powershell
python -c "import zmongo_toolbag, pkgutil; print('ok', bool(pkgutil.find_loader('zmongo_toolbag')))"
```

### 8) (Optional) Upload to **TestPyPI** to try a “real” publish

```powershell
python -m twine upload --repository testpypi .\dist\*
```

### 9) (Optional) Install from **TestPyPI**

```powershell
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple zmongo_retriever==0.1.6
```

### 10) If you saw a “strange location” in the last build log

* That was probably a **TEMP** path used during `pip install .`.
* Force a persistent output using:

```powershell
pip wheel . -w dist
```

Want me to add a `make build` target that runs `python -m build` and a `make publish-test` that runs the `twine upload` flow?
