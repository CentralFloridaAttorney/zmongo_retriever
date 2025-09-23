# ZManager — GUI for ZMongo backup/restore + browse/edit

ZManager is a Tkinter desktop app for managing your MongoDB data via your `ZMongo` repository layer. It lets you:

* Browse collections and scroll through documents
* View a document’s JSON and update **one field** via dot-notation (e.g. `profile.address.city`)
* Insert and delete documents
* Back up and restore collections in **JSON**, **BSON**, or **CSV**
* Use streamlined restore modes: **Merge (Upsert)** or **Replace**&#x20;

---

## 1) Requirements & install

**Python**: 3.10+
**Packages** (typical):

```bash
pip install pymongo motor python-dotenv
```

> Tkinter ships with most Python distributions (on Linux you may need `python3-tk`).
> `bson` comes with PyMongo; do **not** install a standalone `bson` package.

**Project layout assumption**: `zmanager.py` lives alongside your `zmongo_toolbag` package (so `from zmongo_toolbag...` imports work).&#x20;

---

## 2) Environment configuration

ZManager reads the following from your environment (via `~/.resources/.env` if present):

* `MONGO_URI` — default `mongodb://127.0.0.1:27017`
* `MONGO_DATABASE_NAME` — default `test`
* `MONGO_BACKUP_DIR` — relative path under your home (default `.resources/mongo_backups`)

Backups are written to:

```
~/<MONGO_BACKUP_DIR>/<MONGO_DATABASE_NAME>/
```

with filenames like:

```
<collection>[YYYYMMDDHHMMSS].json|bson|csv
```



**Example `.env`:**

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE_NAME=zai_core
MONGO_BACKUP_DIR=.resources/mongo_backups
```

---

## 3) Run it

```bash
python zmanager.py
```

The main window has four tabs:

* **Database Info** — collection list & DB stats (auto-refresh \~30s)
* **Backup & Restore** — pick a collection, choose format, backup/restore
* **Collection Viewer / Editor** — list doc IDs, view JSON, insert/delete, **dot-key update**
* **System Runner** — placeholder for your future tools&#x20;

---

## 4) Database Info tab

Shows:

* Database name
* Collection names
* Object count, data/index/storage sizes

Refreshes periodically.&#x20;

---

## 5) Backup & Restore tab

### Pick a collection

Left pane lists collections. Click one to select. The right pane lists existing backups (by detected filename pattern).

### Choose a format

Dropdown: **JSON**, **BSON**, **CSV**.

* **JSON**: Uses Mongo Extended JSON (via `bson.json_util`) to preserve ObjectIds/dates.
* **BSON**: Writes raw BSON stream (if `BSON`/`decode_file_iter` available); otherwise falls back to JSON for backup and disables BSON restore.
* **CSV**: Flattens docs using **dot-keys** (nested fields become `a.b.c`). Arrays are JSON-encoded in the cell.&#x20;

### Backup actions

* **Backup Selected** — backs up the currently selected collection
* **Backup All** — iterates and backs up all collections

Files are saved as:

```
<collection>[YYYYMMDDHHMMSS].json|bson|csv
```

in your backup directory.&#x20;

### Restore actions

1. Select a file from the **Backup Files** list, or click **Browse for File…**
2. Choose a **Restore Mode**:

   * **Merge (Upsert)** — for each doc: if `_id` exists in DB it’s **replaced**, otherwise it’s **inserted** (non-destructive for docs not in the file).
   * **Replace** — **deletes** all docs in the collection, then inserts the backup’s docs (destructive).
3. Click **Restore Selected**.
   The log shows inserted/matched/modified/upserted counts or any errors.&#x20;

> Notes:
>
> * JSON restore auto-normalizes `"_id"` if it’s a hex string (casts to `ObjectId`).
> * CSV restore **rebuilds nested objects** from dot-keys; values are parsed as JSON when possible, otherwise treated as strings.
> * BSON restore streams from file when supported.&#x20;

---

## 6) Collection Viewer / Editor tab

### Step A — Set collection & filter

* **Collection**: type a collection name (or click “Use Selected (from Backup tab)”)
* **Filter (JSON)**: a MongoDB filter object (default `{}`).
  Examples:

  ```json
  {}
  {"status": "active"}
  {"age": {"$gte": 21}}
  {"_id": {"$in": ["66f2...","66f3..."]}}
  ```
* Click **Refresh** to load the first page (100 IDs).
* Click **Load More** for the next 100 IDs.&#x20;

### Step B — Select a document

* Click an ID in the left list to load the JSON into the right pane.
* The **Document \_id** field in the editor auto-fills.&#x20;

### Step C — Insert or delete

* **Insert Doc**: opens a modal with a JSON editor; submit inserts the document and refreshes the list.
* **Delete Selected**: deletes the currently selected document by `_id`.&#x20;

### Step D — Dot-key update (single doc)

Use the editor at the bottom-right:

* **Document \_id**: Accepts ObjectId hex or string `_id`.
  Click **Use Selected ID** to copy from the current document.
* **Dot-key**: dot-notation path to the field (arrays by index allowed), e.g.:

  * `profile.address.city`
  * `items.0.price`
  * `meta.tags.2`
* **Value**: JSON or plain text.

  * JSON examples: `123`, `true`, `{"a":1}`, `["x","y"]`
  * If parsing as JSON fails, value is stored as a **string**.

Click **Apply \$set** to update.
The log shows `matched_count` / `modified_count`. The JSON view refreshes with the new value.&#x20;

---

## 7) CSV flatten/unflatten semantics

* **Export (backup)**: documents are **flattened** with dot-keys for nested objects (arrays are JSON-encoded in the cell).
* **Import (restore)**: rows are **unflattened** back to nested docs.

  * Keys like `items.0.name` rebuild a list under `items` with index `0`.
  * Each CSV cell attempts `json.loads(value)`; if it fails, it remains a string.
    This allows round-tripping typical shapes, but very complex/irregular documents are best handled via JSON or BSON.&#x20;

---

## 8) Logging & background I/O

* All long operations run on an **async loop** in a background thread; the UI stays responsive.
* The **log panel** at the bottom of *Backup & Restore* shows progress, counts, and errors.
* The app refreshes **Database Info** and **Collections** periodically.&#x20;

---

## 9) Troubleshooting

* **“Cannot import zmongo\_toolbag…”** — ensure your repo is on `PYTHONPATH` or installed in the environment.
* **“BSON streaming not available”** — PyMongo’s `BSON`/`decode_file_iter` not importable; use JSON for backups/restores.
* **ObjectId parsing** — `_id` text that looks like a valid ObjectId hex is auto-cast during restore and when selecting/viewing.
* **CSV import types** — ambiguous scalars may import as strings if not valid JSON. Use JSON/BSON for full fidelity.&#x20;

---

## 10) Keyboard & usability tips

* Use the **Filter (JSON)** to narrow large collections; then page with **Load More**.
* After **Insert**, hit **Refresh** to repopulate the ID list (the app does this for you in most paths).
* Use **Use Selected ID** before dot-key editing to avoid typos in `_id`.&#x20;

---

## 11) Security & safety

* ZManager performs **no schema validation**; it will set any dot-key you specify.
* **Replace** restore will **delete all documents** in the target collection before inserting. Use with care.
* Keep your backups directory secure if documents include sensitive information.&#x20;

---

## 12) Launch checklist

1. Set your `.env` (URI, DB, backup dir).
2. Ensure `zmongo_toolbag` is importable.
3. `python zmanager.py`
4. Pick a collection → **Backup** (choose JSON/BSON/CSV).
5. Use **Collection Viewer / Editor** to browse, insert/delete, and dot-key edit.
6. **Restore** with **Merge (Upsert)** for non-destructive updates, or **Replace** to fully reset a collection.&#x20;

---

**That’s it!** ZManager gives you a simple, dependable MongoDB control panel—fully scrollable views, precise field edits, and robust backup/restore in the formats you actually use.
