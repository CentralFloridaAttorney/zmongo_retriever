
---

## 🛠️ Installing MongoDB and MongoDB Compass on Ubuntu 20.04 / 22.04

This guide walks you through installing MongoDB (version 5.0) and MongoDB Compass—the official GUI—for Ubuntu 20.04 (Focal Fossa) and 22.04 (Jammy Jellyfish). It uses secure practices by importing trusted repository keys and provides clear steps for each phase.

---

### ✅ Step 1: Add MongoDB’s Official GPG Key (Securely)

Ubuntu recommends storing third-party keys in `/usr/share/keyrings`. Use this command to add MongoDB’s signing key securely:

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo tee /usr/share/keyrings/mongodb-archive-keyring.gpg > /dev/null
```

---

### ✅ Step 2: Add the MongoDB Package Repository

Now add the appropriate MongoDB repository based on your Ubuntu version:

- **For Ubuntu 20.04 (Focal):**
  ```bash
  echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
  ```

- **For Ubuntu 22.04 (Jammy):**
  ```bash
  echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
  ```

---

### ✅ Step 3: Install MongoDB

Update your package list and install MongoDB:

```bash
sudo apt update
sudo apt install -y mongodb-org
```

---

### ✅ Step 4: Start and Enable MongoDB Service

Start MongoDB and ensure it launches automatically on boot:

```bash
sudo systemctl start mongod
sudo systemctl enable mongod
```

To verify that MongoDB is running:

```bash
sudo systemctl status mongod
```

You should see `active (running)`.

---

### ✅ Step 5: Install MongoDB Compass (Optional but Recommended)

MongoDB Compass is a graphical interface to explore and manage your MongoDB data.

1. Visit the [MongoDB Compass Download Page](https://www.mongodb.com/try/download/compass)
2. Select:
   - **Platform**: Linux
   - **Package**: `.deb`
   - **Version**: Choose based on your Ubuntu version (20.04 or 22.04)

3. Once downloaded, navigate to your Downloads folder and install it:

```bash
cd ~/Downloads
sudo dpkg -i mongodb-compass-*.deb
```

> 🔧 If you encounter dependency errors, fix them with:
>
> ```bash
> sudo apt --fix-broken install
> ```

You can now launch MongoDB Compass from your applications menu or by typing `mongodb-compass` in the terminal.

---

### 🎉 You're Ready!

You’ve successfully installed MongoDB and MongoDB Compass on Ubuntu. You can now:

- Connect to your MongoDB instance with `mongo` or Compass.
- Use it with applications like `ZMongo`, `LangChain`, and more.
