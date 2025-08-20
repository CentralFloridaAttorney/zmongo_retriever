### 🔹 Create a Pull Request on GitHub (merge `xyzzy` into `master`/`main`)

#### Option A — via the GitHub website

```bash
# make sure your branch is pushed
git push -u origin xyzzy
```

Now in your browser:

1. Go to your repo on GitHub.
2. You’ll see a prompt: “Compare & pull request” for `xyzzy` → click it.
3. Set **base** = `master` (or `main`), **compare** = `xyzzy`.
4. Add a title/description, then click **Create pull request**.
5. (Optional) Request reviewers, address comments, push more commits to `xyzzy` as needed.
6. Click **Merge pull request** → **Confirm merge**.
7. (Optional) **Delete branch** in the UI.

---

#### Option B — via GitHub CLI (`gh`)

```bash
# install gh if you don't have it, then auth:
# Windows: winget install GitHub.cli
# macOS:   brew install gh
# Linux:   sudo apt-get install gh   (or your package manager)
gh auth login

# from your repo, ensure branch is pushed
git push -u origin xyzzy

# create PR to master (change to 'main' if needed)
gh pr create --base master --head xyzzy --title "Merge xyzzy" --body "Details of the change"

# open the PR in browser (review/merge there)
gh pr view --web

# OR merge from CLI (squash example)
gh pr merge --squash --delete-branch
```

> After the PR is merged, pull the updated default branch locally:

```bash
git checkout master-081925   # or: git checkout main
git pull
```
