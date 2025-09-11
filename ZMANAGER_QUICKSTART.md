
# ZManager: MongoDB GUI Assistant

ZManager is a desktop graphical user interface (GUI) built with Python's Tkinter library to provide a simple, powerful way to manage and interact with your MongoDB databases. It is designed to work seamlessly with the `ZMongo` library, offering features for browsing data, editing documents, and performing robust backup and restore operations.



This tool is perfect for developers and database administrators who need a straightforward way to view and manipulate MongoDB data without writing complex scripts for common tasks.

## Table of Contents
1.  [Features](#features)
2.  [Prerequisites & Setup](#prerequisites--setup)
3.  [How to Run ZManager](#how-to-run-zmanager)
4.  [Using the Application](#using-the-application)
    * [Tab 1: Database Info](#tab-1-database-info)
    * [Tab 2: Backup & Restore](#tab-2-backup--restore)
    * [Tab 3: Collection Viewer / Editor](#tab-3-collection-viewer--editor)
5.  [Troubleshooting](#troubleshooting)

---

## Features

* **Database Overview:** At-a-glance view of all collections and key database statistics (document count, data size, etc.).
* **Collection Browser:** Easily list and view documents within any collection, with support for pagination ("Load More").
* **Document Viewer:** View the full JSON content of any selected document in a clear, formatted display.
* **Live Document Editor:** Perform precise updates on any document using dot-notation to target specific fields (e.g., `user.profile.settings.darkMode`).
* **Document Management:** Insert new JSON documents or delete existing ones directly from the UI.
* **Flexible Backups:**
    * Backup a single collection or the entire database.
    * Choose from multiple formats: **JSON**, **BSON**, or **CSV**.
* **Robust Restore:**
    * Restore a collection from a backup file.
    * **Merge (Upsert) Mode:** Add new documents and update existing ones (based on `_id`).
    * **Replace Mode:** Wipe the collection and replace its content entirely with the backup.

---

## Prerequisites & Setup

Before running ZManager, ensure you have the following set up.

### 1. Python Environment
Make sure you have a working Python environment with all the necessary libraries installed. You can install them using pip:
```bash
pip install .
````

### 2\. MongoDB Connection

ZManager connects to your MongoDB instance using environment variables. You must create a file to store these settings.

Create a file named `.env_zai_core` in the `.resources` directory of your user's home folder.

  * **Windows:** `C:\Users\YourUser\.resources\.env_zai_core`
  * **macOS/Linux:** `/Users/YourUser/.resources/.env_zai_core`

Add the following lines to this file, adjusting the values to match your MongoDB setup:

```env
# The connection string for your MongoDB server
MONGO_URI="mongodb://127.0.0.1:27017"

# The name of the database you want to manage
MONGO_DATABASE_NAME="your_database_name"
```

-----

## How to Run ZManager

To launch the application, simply run the `zmanager.py` script from your terminal:

```bash
python zmongo_toolbag\zmanager.py
```

The application window will appear, ready for you to use.

-----

## Using the Application

The application is organized into three main tabs, each designed for a specific set of tasks.

### Tab 1: Database Info

This is the main dashboard. It provides a read-only overview of your database.

  * **Collections List:** A list of all collections currently in your database.
  * **DB Stats:** Key metrics like the total number of documents and the size of your data and indexes on disk.
  * **Auto-Refresh:** This information automatically refreshes every 30 seconds.

### Tab 2: Backup & Restore

This tab provides all the tools you need to create and restore backups.

#### How to Create a Backup:

1.  **Select a Collection:** Click on a collection name from the list on the left. It will appear in the "Selected Collection" box.
2.  **Choose a Format:** Select JSON, BSON, or CSV from the "Backup format" dropdown.
3.  **Click Backup:**
      * Click **`Backup Selected`** to back up only the collection you chose.
      * Click **`Backup All`** to create separate backup files for every collection in the database.
4.  **Confirmation:** A message will appear in the log panel at the bottom confirming the backup was successful. Backup files are saved to the `.resources/mongo_backups` directory in your home folder.

#### How to Restore from a Backup:

1.  **Select Target Collection:** Click on the collection you want to restore data into.
2.  **Select Backup File:** The "Backup Files" list automatically shows files corresponding to the selected collection. Click the one you want to restore.
      * Alternatively, click **`Browse for File...`** to select a backup file from anywhere on your computer.
3.  **Choose Restore Mode:**
      * **Merge (Upsert):** This is the safest option. It will update existing documents (matching by `_id`) and insert any new documents from the backup file.
      * **Replace:** **(Use with caution\!)** This will completely delete all data in the target collection before inserting the documents from the backup file.
4.  **Click Restore:** Click the **`Restore Selected`** button.
5.  **Confirmation:** Check the log panel for a summary of the restore operation (e.g., number of documents inserted, modified, etc.).

### Tab 3: Collection Viewer / Editor

This powerful tab allows you to browse, view, and edit individual documents.

#### Browsing Documents:

1.  **Enter Collection Name:** Type the name of a collection into the "Collection" entry box. You can use the **`Use Selected`** button to auto-fill it from the Backup tab.
2.  **Apply Filter (Optional):** Enter a valid MongoDB JSON filter in the "Filter" box (e.g., `{"topic": "Biology"}`). Leave it as `{}` to see all documents.
3.  **Click Refresh:** A list of document `_id`s will appear in the "Documents" listbox on the left.
4.  **Load More:** If the collection has more documents than the page limit (100), click **`Load More`** to append the next page of results.

#### Viewing and Editing a Document:

1.  **Select a Document:** Click on any `_id` in the "Documents" list. The full JSON content of that document will appear in the large text panel on the right.
2.  **Use the Dot-Key Editor:**
      * The document's `_id` will be automatically filled in the `Document _id` field.
      * In the **`Dot-key`** field, enter the path to the field you want to change (e.g., `metadata.status` or `items.0.name`).
      * In the **`Value`** field, enter the new value. This can be a simple string, a number, or valid JSON (e.g., `true`, `["a", "b"]`, or `{"x": 1}`).
      * Click **`Apply $set`**.
3.  **Confirmation:** A success message will appear in the log panel of the Backup tab. The JSON view will automatically refresh to show the updated document.

#### Inserting and Deleting Documents:

  * **Insert:** Click the **`Insert Doc`** button. A new window will appear where you can paste the JSON for a new document.
  * **Delete:** Select a document from the list and click the **`Delete Selected`** button.

-----

## Troubleshooting

  * **Connection Failed:** Ensure your `MONGO_URI` in the `.env_zai_core` file is correct and that your MongoDB server is running.
  * **Invalid JSON Filter:** If you get an error when refreshing documents, double-check that your filter string is valid JSON (e.g., keys and strings must be in double-quotes).
  * **BSON Errors:** The BSON backup/restore format requires specific library versions (`pymongo>=4.0`). If it fails, the application will automatically fall back to using JSON, which is universally compatible.

<!-- end list -->

```
