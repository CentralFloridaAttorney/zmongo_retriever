# Makefile for zmongo_retriever

.PHONY: help install dev test lint docs clean

help:
	@echo "Available targets:"
	@echo "  install   - Install runtime dependencies"
	@echo "  dev       - Install dev + test + lint + docs extras"
	@echo "  test      - Run pytest with coverage"
	@echo "  lint      - Run ruff, mypy, black, and isort checks"
	@echo "  docs      - Build Sphinx docs into docs/build/html"
	@echo "  clean     - Remove build, dist, cache, and docs output"

install:
	pip install .

dev:
	pip install -e .[dev,test,lint,docs]

test:
	pytest --cov=zmongo_toolbag --cov-report=term-missing -q

lint:
	ruff check src tests
	mypy src
	black --check src tests
	isort --check-only src tests

docs:
	python -m sphinx -b html docs/source docs/build/html
	@echo "Docs available at docs/build/html/index.html"

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache docs/build
	find . -type d -name "__pycache__" -exec rm -rf {} +
