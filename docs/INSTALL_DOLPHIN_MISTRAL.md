# How to Download the Dolphin 2.2.1 Mistral 7B-GGUF Model

This guide provides instructions on how to download the Dolphin 2.2.1 Mistral 7B-GGUF model hosted on Hugging Face. Before you proceed, ensure you have Git and Git Large File Storage (LFS) installed on your machine. Git LFS is necessary for handling the large files associated with the model.

## Prerequisites

- **Git**: Git must be installed to clone the repository. If you do not have Git installed, you can download it from [here](https://git-scm.com/downloads).
- **Git LFS**: Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server. Install Git LFS by following the instructions on the [official website](https://git-lfs.github.com).

## Step-by-Step Instructions

### 1. Install Git LFS

If you haven't installed Git LFS, open your terminal or command prompt and execute the following command:

```sh
git lfs install
```

This command sets up Git LFS for your user account.

### 2. Clone the Model Repository

To download the model with all the large files, run:

```sh
git clone https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
```

This command clones the repository to your local machine, downloading all the files, including the large ones managed by Git LFS.

### Optional: Clone Without Large Files

If you prefer to clone the repository without downloading the large files immediately (downloading only their pointers instead), you can use the following command:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
```

You can later fetch the large files on demand using Git LFS commands.

## After Downloading

Once the repository is cloned to your local machine, you will find the Dolphin 2.2.1 Mistral 7B-GGUF model files within the cloned directory. You can now proceed to use the model as per your requirements.

For more information on using the model or troubleshooting any issues, consider visiting the model's page on Hugging Face or the Git and Git LFS documentation.