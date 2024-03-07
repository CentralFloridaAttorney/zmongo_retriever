To install MongoDB on Ubuntu Linux (versions 20.04 and 22.04) and MongoDB Compass, follow the step-by-step guide below. This guide incorporates the use of trusted keys for repository security and provides instructions for selecting the correct operating system parameters when downloading MongoDB Compass.

### Step 1: Import the MongoDB Repository GPG Key
Ubuntu now recommends adding repository keys to a trusted key directory instead of using `apt-key`. Follow these steps to import the MongoDB GPG key:

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo tee /usr/share/keyrings/mongodb-archive-keyring.gpg >/dev/null
```

### Step 2: Add the MongoDB Repository
Create a `.list` file in the `/etc/apt/sources.list.d/` directory. This file tells apt where to fetch MongoDB packages.

- For Ubuntu 20.04 (Focal Fossa):

```bash
echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
```

- For Ubuntu 22.04 (Jammy Jellyfish):

```bash
echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
```

### Step 3: Install MongoDB
Update your local package database and install MongoDB with the following commands:

```bash
sudo apt-get update
sudo apt-get install -y mongodb-org
```

### Step 4: Start MongoDB and Enable Auto-Start
Enable MongoDB to start on boot and then start the MongoDB service:

```bash
sudo systemctl enable mongod
sudo systemctl start mongod
```

### Install MongoDB Compass
MongoDB Compass is a GUI that allows you to analyze your database visually. To install MongoDB Compass, follow the link below to the MongoDB download center and select the appropriate options for your operating system (Linux), distribution version (Ubuntu 20.04 or 22.04), and package format (`.deb`).

[MongoDB Compass Download](https://www.mongodb.com/try/download/compass)

After downloading the `.deb` package, navigate to your download directory and install MongoDB Compass using the following command:

```bash
sudo dpkg -i mongodb-compass-*.deb
```
Make sure to replace `mongodb-compass-*.deb` with the actual file name of the downloaded package.

MongoDB Compass can now be launched from your applications menu or by executing `mongodb-compass` in your terminal.

These instructions ensure that you are using the correct and secure method to install MongoDB and MongoDB Compass on Ubuntu, following best practices for repository security.