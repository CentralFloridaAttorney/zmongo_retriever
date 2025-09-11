# -----------------------------
# Project Makefile (Windows-friendly, cmd.exe shell)
# -----------------------------

SHELL := cmd
.SHELLFLAGS := /C

# Tools (override like: make VAR=value ...)
PY      ?= python
PIP     ?= pip
RUFF    ?= ruff
PYTEST  ?= pytest
MYPY    ?= mypy

# Project dirs
PKG_DIRS   ?= zmongo_toolbag
TEST_DIR   ?= tests
LINT_DIRS  ?= $(PKG_DIRS) $(TEST_DIR)

# Default goal
.DEFAULT_GOAL := help

# -----------------------------
# Help
# -----------------------------
.PHONY: help
help:
	@echo Common targets:
	@echo   make lint        - Run Ruff lint on $(LINT_DIRS)
	@echo   make fix         - Ruff auto-fix and format
	@echo   make format      - Ruff formatter only
	@echo   make test        - Run tests with pytest
	@echo   make cov         - Run tests with coverage
	@echo   make typecheck   - Run mypy on $(PKG_DIRS)
	@echo   make build       - Build wheel/sdist
	@echo   make clean       - Remove build, cache, and temp files
	@echo   make check       - lint + typecheck + tests

# -----------------------------
# Lint / Format
# -----------------------------
.PHONY: lint
lint:
	$(RUFF) check $(LINT_DIRS)

.PHONY: fix
fix:
	$(RUFF) check --fix $(LINT_DIRS)
	$(RUFF) format $(LINT_DIRS)

.PHONY: format
format:
	$(RUFF) format $(LINT_DIRS)

# CI-friendly output (e.g., GitHub Actions annotations)
.PHONY: lint-ci
lint-ci:
	$(RUFF) check --output-format=github $(LINT_DIRS)

# -----------------------------
# Tests / Coverage
# -----------------------------
.PHONY: test
test:
	$(PYTEST) -q

.PHONY: cov
cov:
	$(PYTEST) --maxfail=1 --disable-warnings --cov=$(PKG_DIRS) --cov-report=term-missing

# -----------------------------
# Typing
# -----------------------------
.PHONY: typecheck
typecheck:
	$(MYPY) $(PKG_DIRS)

# -----------------------------
# Build
# -----------------------------
.PHONY: build
build:
	$(PY) -m build

# -----------------------------
# Aggregate checks
# -----------------------------
.PHONY: check
check:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

# -----------------------------
# Clean (cross-platform via single-line Python -c)
# -----------------------------
.PHONY: clean
clean:
	-$(PY) -c "import os,shutil,glob; \
paths=['build','dist','.pytest_cache','.mypy_cache','.ruff_cache']; \
paths+=glob.glob('*.egg-info'); \
paths+=glob.glob('**\\\\__pycache__', recursive=True); \
[shutil.rmtree(p, ignore_errors=True) for p in paths]; \
print('Cleaned.')"
