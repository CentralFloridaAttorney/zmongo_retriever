---

# 🐬 How to Download the Dolphin 2.2.1 Mistral 7B-GGUF Model

Need a powerful local LLM? This guide will walk you through downloading the **Dolphin 2.2.1 Mistral 7B-GGUF** model hosted on 🤗 Hugging Face.

Before you begin, make sure **Git** and **Git LFS** (Large File Storage) are installed — these tools are essential for downloading models with large file sizes.

---

## ✅ Prerequisites

You'll need the following installed:

- **🔧 Git**  
  👉 [Download Git](https://git-scm.com/downloads)

- **📦 Git LFS (Large File Storage)**  
  👉 [Install Git LFS](https://git-lfs.github.com)

Git LFS ensures that large model files are downloaded properly without corrupting your clone.

---

## 🧭 Step-by-Step Instructions

### 🧰 1. Install Git LFS

Open your terminal and run:

```bash
git lfs install
```

> ✅ This command sets up Git LFS support for your user account.

---

### 📥 2. Clone the Dolphin Model Repository

Use the following command to clone the **entire repository**, including large `.gguf` model files:

```bash
git clone https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
```

This will download the model into a folder called `dolphin-2.2.1-mistral-7B-GGUF`.

---

### 💡 Optional: Clone Without Downloading Large Files

Want to preview or manually choose files before downloading all the large model weights? You can **skip downloading** LFS files initially:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
```

Then fetch specific files later using:

```bash
git lfs pull
```

---

## 📂 After Cloning

Once downloaded, the directory will contain all the necessary `.gguf` model files.

You can now:

- Load it with tools like `llama.cpp`, `text-generation-webui`, or `llamacpp-python`
- Reference it in scripts via its full path
- Explore Hugging Face for **README usage tips** or **performance benchmarks**

---

## 📚 Helpful Links

- 🔗 [Model on Hugging Face](https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF)  
- 🧰 [Git LFS Docs](https://git-lfs.github.com)  
- 🛠 [GitHub LFS Troubleshooting](https://github.com/git-lfs/git-lfs/wiki/Troubleshooting)

---

## ✅ You're Ready to Go!

That’s it! You now have everything you need to start using **Dolphin 2.2.1 Mistral 7B-GGUF** locally.
