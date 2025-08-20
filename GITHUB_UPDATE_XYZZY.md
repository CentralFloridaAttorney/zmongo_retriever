Here’s the clean sequence to create a new branch **xyzzy**, switch to it, and push it to GitHub:

```bash
# 1. Make sure your working directory is clean
git status

# 2. Create and switch to the new branch
git checkout -b xyzzy

# 3. Do your work, then stage changes
git add .

# 4. Commit your changes
git commit -m "Work on xyzzy branch"

# 5. Push the branch to GitHub and set upstream
git push -u origin xyzzy
```
