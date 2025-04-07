

### ✅ `CONTRIBUTING.md`

```markdown
# Contributing to `zmongo_retriever`

Thank you for considering a contribution to this project!

We welcome issues, feature suggestions, test improvements, and pull requests — especially those that improve security, documentation, and developer experience.

---

## 🧭 Project Status

This project is currently in **pre-release** and under active development.  
Production usage is not yet recommended.

Before contributing, please read our:

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

---

## 🧰 Local Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CentralFloridaAttorney/zmongo_retriever.git
   cd zmongo_retriever
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -e .[dev]
   ```

---

## 📁 Directory Overview

- `zmongo_retriever/` – core logic and interfaces
- `tests/` – unit tests
- `.env.example` – sample environment variables
- `setup.py` – Python packaging entry point

---

## 🧪 Running Tests

Tests are located in the root-level `/tests` folder.

To run them:

```bash
python -m unittest discover -s tests
```

Or with `pytest` (if available):

```bash
pytest tests/
```

Before opening a pull request, **all tests must pass**.

---

## 🔃 Git Workflow

1. Fork this repo and clone your fork locally.
2. Create a new branch for your contribution:

   ```bash
   git checkout -b feature/add-caching-layer
   ```

3. Make your changes and commit clearly:

   ```bash
   git commit -m "feat: add chunk caching to ZRetriever"
   ```

4. Push to your fork:

   ```bash
   git push origin feature/add-caching-layer
   ```

5. Open a pull request on GitHub targeting the `main` branch.

---

## 🧠 Guidelines

- Keep pull requests small and focused.
- If adding new modules or integrations (e.g. Claude, Ollama, new DBs), write tests.
- Include descriptive commit messages and PR summaries.
- Document environment variables or setup changes clearly.
- Avoid hardcoded secrets, keys, or tokens.

---

## 💬 Communication

For significant changes, questions, or proposals, please open a [Discussion](https://github.com/CentralFloridaAttorney/zmongo_retriever/discussions) or email us:

📬 [Contact@CentralFloridaAttorney.net](mailto:Contact@CentralFloridaAttorney.net)  
📬 [Attorney@CentralFloridaAttorney.net](mailto:Attorney@CentralFloridaAttorney.net)

---

Thank you for helping make `zmongo_retriever` better for everyone!
