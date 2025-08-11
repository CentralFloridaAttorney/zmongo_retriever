---

# 🔐 Managing Your OpenAI API Key

Welcome to your quick-start guide for safely storing, accessing, and using your **OpenAI API key**. Keeping this key secure is critical to protecting your billing, data, and AI usage.

---

## 📥 How to Get Your API Key

1. Go to [OpenAI](https://openai.com/api/).
2. Sign up or log in to your OpenAI account.
3. Navigate to the **API Keys** section in your account settings.
4. Click **Create new secret key** and copy the key shown.  
   > ⚠️ **You won’t be able to view it again!** Save it securely.

---

## 🔐 Where to Store the Key (Securely)

### ✅ Option 1: Use Environment Variables (Recommended)

#### 🪟 Windows
- Open **System Properties** → **Environment Variables**.
- Create a new **User variable**:  
  `OPENAI_API_KEY = your-api-key`

#### 🍎 macOS / 🐧 Linux
Add this line to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export OPENAI_API_KEY="your-api-key"
```

Then reload your shell:

```bash
source ~/.zshrc  # or ~/.bashrc, depending on your shell
```

---

### ✅ Option 2: Use a `.env` File

1. In your project root, create a file named `.env`:

```bash
touch .env
```

2. Add your key:

```
OPENAI_API_KEY=your-api-key
```

3. Make sure your `.env` file is ignored by Git:

```
# .gitignore
.env
```

---

## 🧠 How to Use the Key in Python

Use this snippet to securely load your API key in your scripts:

```python
import os
from dotenv import load_dotenv

load_dotenv(from pathlib import Pathload_dotenv(Path.home() / "resources" / ".env"))  # Only needed if you're using a .env file

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")
```

---

## 🔒 Security Best Practices

- ✅ **Never** hard-code your API key in Python scripts or notebooks.
- ✅ Use `.env` files **only locally** — never commit them to GitHub.
- 🔄 Rotate your API key periodically.
- 🧾 Monitor your OpenAI usage for unusual activity.
- 🔐 Use **role-based access** and **rate limits** where available.

---

## 📚 Additional Resources

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Security Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)

---

## 🏁 Final Thoughts

Your API key is your passport to OpenAI’s powerful tools—treat it like a password. A few minutes of setup goes a long way in keeping your apps secure and production-ready. ✨
