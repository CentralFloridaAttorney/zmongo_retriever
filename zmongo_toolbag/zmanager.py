# zmanager.py
import asyncio
import csv
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, Tk, Listbox, Entry, Button, Frame, Toplevel, END, BOTH, LEFT, RIGHT, Y, X, TOP
from tkinter.scrolledtext import ScrolledText

from bson import errors
from bson.objectid import ObjectId
from bson import json_util

try:
    # Available in PyMongo >= 4.x
    from bson import BSON, decode_file_iter

    HAVE_BSON_STREAM = True
except Exception:
    HAVE_BSON_STREAM = False

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, InsertOne, ReplaceOne
from pymongo.errors import BulkWriteError

from zmongo_toolbag.data_processing import DataProcessor, SafeResult
from zmongo_toolbag.zmongo import ZMongo

# --- Configuration and Setup ---
load_dotenv(Path.home() / ".resources" / ".env_zai_core")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables with sensible defaults
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "test")
# Default backup directory relative to home
MONGO_BACKUP_DIR_REL = os.getenv("MONGO_BACKUP_DIR", '.resources/mongo_backups')


@dataclass
class Pager:
    limit: int = 100
    skip: int = 0


class ZManager(Tk):
    """
    Tkinter GUI for ZMongo management:
    - Browse collections & documents (scrollable)
    - Dot-key single-document updates
    - Backup/Restore in JSON/BSON/CSV
    - Insert/Delete documents
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.zmongo = ZMongo()
        self.title("ZMongo System Manager")
        self.geometry("1400x900")

        self.loop = loop
        self.db_name = MONGO_DATABASE_NAME

        # THE FIX: Define backup_dir as a Path object from the start.
        self.backup_dir = Path.home() / MONGO_BACKUP_DIR_REL / MONGO_DATABASE_NAME
        # The make_dir_if_not_exists call is now simpler.
        self.make_dir_if_not_exists(self.backup_dir)

        # MongoDB clients (async + sync)
        try:
            self.async_client = AsyncIOMotorClient(MONGO_URI)
            self.db = self.async_client[self.db_name]
            self.sync_client = MongoClient(MONGO_URI)
            self.sync_db = self.sync_client[self.db_name]
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            self.destroy()
            return

        # State for Collection Viewer
        self.cv_pager = Pager(limit=100, skip=0)
        self.cv_active_collection = ""
        self.cv_filter_text = "{}"
        self.cv_ids_cache = []  # list of (_id_str, _id_obj)

        self._create_widgets()
        self.run_periodic_updates()

    # ---------- UI BUILD ----------

    def _create_widgets(self):
        main_notebook = ttk.Notebook(self)
        main_notebook.pack(expand=True, fill=BOTH, padx=10, pady=10)

        db_info_tab = ttk.Frame(main_notebook)
        maintenance_tab = ttk.Frame(main_notebook)
        collection_tab = ttk.Frame(main_notebook)
        system_tab = ttk.Frame(main_notebook)

        main_notebook.add(db_info_tab, text='Database Info')
        main_notebook.add(maintenance_tab, text='Backup & Restore')
        main_notebook.add(collection_tab, text='Collection Viewer / Editor')
        main_notebook.add(system_tab, text='System Runner')

        # --- Database Info Tab ---
        self.db_info_text = ScrolledText(db_info_tab, wrap="word", font=("Courier New", 10))
        self.db_info_text.pack(expand=True, fill=BOTH, padx=5, pady=5)

        # --- Maintenance Tab (Backup/Restore) ---
        self._build_maintenance_tab(maintenance_tab)

        # --- Collection Viewer / Editor Tab ---
        self._build_collection_tab(collection_tab)

        # --- System tab placeholder (optional future runners) ---
        st = ScrolledText(system_tab, height=8, wrap="word")
        st.pack(expand=True, fill=BOTH, padx=8, pady=8)
        st.insert("end", "System Runner: (reserve for future services)\n")

    def _build_maintenance_tab(self, parent: Frame):
        root = ttk.Frame(parent)
        root.pack(fill=BOTH, expand=True, padx=8, pady=8)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(3, weight=1)
        root.grid_rowconfigure(5, weight=1)

        ttk.Label(root, text="Collections:").grid(row=0, column=0, sticky="w", padx=5)
        listbox_frame = ttk.Frame(root)
        listbox_frame.grid(row=1, column=0, rowspan=4, sticky="nswe", padx=5)
        self.collection_listbox = Listbox(listbox_frame, exportselection=False, height=12)
        self.collection_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        sb1 = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.collection_listbox.yview)
        sb1.pack(side=RIGHT, fill=Y)
        self.collection_listbox.config(yscrollcommand=sb1.set)
        self.collection_listbox.bind('<<ListboxSelect>>', self.on_collection_select)

        ttk.Label(root, text="Selected Collection:").grid(row=0, column=1, sticky="w", padx=5)
        self.selected_collection_entry = Entry(root, state='readonly')
        self.selected_collection_entry.grid(row=1, column=1, sticky="we", padx=5)

        ttk.Label(root, text="Backup Files:").grid(row=0, column=2, sticky="w", padx=5)
        backup_frame = ttk.Frame(root)
        backup_frame.grid(row=1, column=2, rowspan=4, sticky="nswe", padx=5)
        self.backup_files_listbox = Listbox(backup_frame, exportselection=False, height=12)
        self.backup_files_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        sb2 = ttk.Scrollbar(backup_frame, orient="vertical", command=self.backup_files_listbox.yview)
        sb2.pack(side=RIGHT, fill=Y)
        self.backup_files_listbox.config(yscrollcommand=sb2.set)
        self.backup_files_listbox.bind('<<ListboxSelect>>', self.on_backup_file_select)

        ttk.Label(root, text="Selected Backup File:").grid(row=0, column=3, sticky="w", padx=5)
        self.selected_backup_entry = Entry(root, state='readonly')
        self.selected_backup_entry.grid(row=1, column=3, sticky="we", padx=5)

        # Backup format & actions
        actions = ttk.Frame(root)
        actions.grid(row=2, column=1, columnspan=3, sticky="we", pady=10)
        ttk.Label(actions, text="Backup format:").pack(side=LEFT, padx=(0, 6))
        self.backup_format_combo = ttk.Combobox(actions, state='readonly', values=["JSON", "BSON", "CSV"], width=8)
        self.backup_format_combo.current(0)
        self.backup_format_combo.pack(side=LEFT, padx=(0, 12))

        Button(actions, text='Backup Selected', command=self.on_backup_selected_clicked).pack(side=LEFT, padx=5)
        Button(actions, text='Backup All', command=self.on_backup_all_clicked).pack(side=LEFT, padx=5)
        Button(actions, text='Restore Selected', command=self.on_restore_clicked).pack(side=LEFT, padx=5)
        Button(actions, text='Browse for File...', command=self.open_file_explorer).pack(side=LEFT, padx=5)

        # Restore mode simplified
        self.restore_options = ttk.Combobox(root, state='readonly', values=[
            "Merge (Upsert)", "Replace"
        ])
        self.restore_options.current(0)
        self.restore_options.grid(row=3, column=1, columnspan=3, sticky="we", padx=5)

        # Log
        self.message_text = ScrolledText(root, height=8, wrap="word", state='disabled')
        self.message_text.grid(row=5, column=0, columnspan=4, sticky="nswe", padx=5, pady=5)

    def _build_collection_tab(self, parent: Frame):
        # Two-pane: Left = document list; Right = details & editor
        root = ttk.Frame(parent)
        root.pack(fill=BOTH, expand=True, padx=8, pady=8)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)

        # Row 0: collection name + controls
        ttk.Label(root, text="Collection:").grid(row=0, column=0, sticky="w")
        self.cv_collection_entry = Entry(root, width=40)
        self.cv_collection_entry.grid(row=0, column=1, sticky="w", padx=6)
        # filter
        ttk.Label(root, text="Filter (JSON):").grid(row=1, column=0, sticky="w")
        self.cv_filter_entry = Entry(root)
        self.cv_filter_entry.insert(0, "{}")
        self.cv_filter_entry.grid(row=1, column=1, sticky="we", padx=6)

        ctrl = ttk.Frame(root)
        ctrl.grid(row=0, column=2, rowspan=2, sticky="ne")
        Button(ctrl, text="Use Selected (from Backup tab)", command=self._prefill_collection_from_tab2).pack(side=TOP,
                                                                                                             padx=4,
                                                                                                             pady=2)
        Button(ctrl, text="Refresh", command=self.cv_refresh_docs_clicked).pack(side=TOP, padx=4, pady=2)
        Button(ctrl, text="Load More", command=self.cv_load_more_clicked).pack(side=TOP, padx=4, pady=2)

        # Row 2: Main panes
        left = ttk.Frame(root, borderwidth=1, relief="groove")
        left.grid(row=2, column=0, sticky="ns")
        left.grid_rowconfigure(1, weight=1)

        ttk.Label(left, text="Documents").grid(row=0, column=0, sticky="we")
        doc_list_frame = ttk.Frame(left)
        doc_list_frame.grid(row=1, column=0, sticky="ns")
        self.cv_doc_listbox = Listbox(doc_list_frame, height=30, width=36, exportselection=False)
        self.cv_doc_listbox.pack(side=LEFT, fill=Y, expand=False)
        sb_docs = ttk.Scrollbar(doc_list_frame, orient="vertical", command=self.cv_doc_listbox.yview)
        sb_docs.pack(side=RIGHT, fill=Y)
        self.cv_doc_listbox.config(yscrollcommand=sb_docs.set)
        self.cv_doc_listbox.bind('<<ListboxSelect>>', self.cv_on_doc_select)

        # Insert/Delete controls below doc list
        actions = ttk.Frame(left)
        actions.grid(row=2, column=0, sticky="we", pady=(8, 0))
        Button(actions, text="Insert Doc", command=self.cv_insert_doc_dialog).pack(side=LEFT, padx=4)
        Button(actions, text="Delete Selected", command=self.cv_delete_selected).pack(side=LEFT, padx=4)

        # Right pane: details and dot-key editor
        right = ttk.Frame(root, borderwidth=1, relief="groove")
        right.grid(row=2, column=1, columnspan=2, sticky="nswe", padx=(8, 0))
        right.grid_columnconfigure(1, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ttk.Label(right, text="Selected Document JSON").grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.cv_json_text = ScrolledText(right, font=("Courier New", 10), wrap="none")
        self.cv_json_text.grid(row=1, column=0, columnspan=2, sticky="nswe", padx=6, pady=6)

        # Dot-key editor (single doc)
        editor = ttk.LabelFrame(right, text="Dot-Key Update (single doc)")
        editor.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 8))
        editor.grid_columnconfigure(1, weight=1)

        ttk.Label(editor, text="Document _id:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.cv_id_entry = Entry(editor)
        self.cv_id_entry.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        Button(editor, text="Use Selected ID", command=self._use_selected_id).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(editor, text="Dot-key (e.g., a.b.0.c):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.cv_dotkey_entry = Entry(editor)
        self.cv_dotkey_entry.grid(row=1, column=1, sticky="we", padx=5, pady=5)

        ttk.Label(editor, text="Value (JSON or text):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.cv_value_entry = Entry(editor)
        self.cv_value_entry.grid(row=2, column=1, sticky="we", padx=5, pady=5)

        Button(editor, text="Apply $set", command=self.on_apply_dotkey_value_clicked).grid(row=3, column=0,
                                                                                           columnspan=3, sticky="we",
                                                                                           padx=6, pady=(6, 8))

    # ---------- Helpers ----------

    def _prefill_collection_from_tab2(self):
        selected = self.selected_collection_entry.get().strip()
        if selected:
            self.cv_collection_entry.delete(0, END)
            self.cv_collection_entry.insert(0, selected)

    def _use_selected_id(self):
        idxs = self.cv_doc_listbox.curselection()
        if not idxs:
            self.log_message("No document selected.")
            return
        _id_str, _ = self.cv_ids_cache[idxs[0]]
        self.cv_id_entry.delete(0, END)
        self.cv_id_entry.insert(0, _id_str)

    @staticmethod
    def _parse_input_value(raw: str):
        s = (raw or "").strip()
        if not s:
            return ""
        try:
            return json.loads(s)
        except Exception:
            return s

    @staticmethod
    def _parse_objectid(raw: str):
        s = (raw or "").strip()
        if not s:
            return None
        try:
            return ObjectId(s)
        except Exception:
            return None

    @staticmethod
    def _flatten_for_csv(doc: dict) -> dict:
        return DataProcessor.flatten_json(doc)

    @staticmethod
    def _unflatten_from_csv(row: dict) -> dict:
        root = {}

        def set_path(container, parts, value):
            if not parts:
                return value
            key = parts[0]
            is_index = key.isdigit()

            if is_index:
                index = int(key)
                if not isinstance(container, list):
                    container_ref = []
                else:
                    container_ref = container
                while len(container_ref) <= index:
                    container_ref.append({})
                container_ref[index] = set_path(container_ref[index], parts[1:], value)
                return container_ref
            else:
                if not isinstance(container, dict):
                    container_ref = {}
                else:
                    container_ref = container
                container_ref[key] = set_path(container_ref.get(key, {}), parts[1:], value)
                return container_ref

        for k, v in row.items():
            if v is None or v == "":
                continue
            try:
                parsed = json.loads(v)
            except Exception:
                parsed = v
            parts = k.split(".")
            root = set_path(root, parts, parsed)

        return root

    def run_in_async_loop(self, async_func, *args, **kwargs):
        future = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), self.loop)
        future.add_done_callback(self.on_async_task_done)
        return future

    def on_async_task_done(self, future):
        try:
            future.result()
        except Exception as e:
            logging.error(f"Async task failed: {e}")
            self.log_message(f"Error: {e}")

    def log_message(self, message):
        def _append():
            self.message_text.config(state='normal')
            self.message_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
            self.message_text.config(state='disabled')
            self.message_text.see("end")

        self.after(0, _append)

    @staticmethod
    def make_dir_if_not_exists(directory: Path):
        directory.mkdir(parents=True, exist_ok=True)

    # ---------- Event Handlers (Backup/Restore UI) ----------

    def on_collection_select(self, event=None):
        selection = self.collection_listbox.curselection()
        if not selection:
            return
        collection_name = self.collection_listbox.get(selection[0])
        self.selected_collection_entry.config(state='normal')
        self.selected_collection_entry.delete(0, "end")
        self.selected_collection_entry.insert(0, collection_name)
        self.selected_collection_entry.config(state='readonly')
        self.update_backup_files_listbox(collection_name)
        self.log_message(f"Selected collection: {collection_name}")

    def on_backup_file_select(self, event=None):
        selection = self.backup_files_listbox.curselection()
        if not selection:
            return
        filename = self.backup_files_listbox.get(selection[0])
        self.selected_backup_entry.config(state='normal')
        self.selected_backup_entry.delete(0, "end")
        self.selected_backup_entry.insert(0, filename)
        self.selected_backup_entry.config(state='readonly')
        self.log_message(f"Selected backup file: {filename}")

    def on_backup_selected_clicked(self):
        collection_name = self.selected_collection_entry.get().strip()
        if not collection_name:
            self.log_message("Error: No collection selected for backup.")
            return
        fmt = self.backup_format_combo.get()
        self.log_message(f"Starting {fmt} backup for '{collection_name}'...")
        self.run_in_async_loop(self.backup_collection, collection_name, fmt)

    def on_backup_all_clicked(self):
        fmt = self.backup_format_combo.get()
        self.log_message(f"Starting {fmt} backup for all collections...")
        self.run_in_async_loop(self.backup_all_collections, fmt)

    def on_restore_clicked(self):
        collection_name = self.selected_collection_entry.get().strip()
        backup_file_or_path = self.selected_backup_entry.get().strip()
        restore_mode = self.restore_options.get()

        if not collection_name and backup_file_or_path:
            collection_name = Path(backup_file_or_path).name.partition('[')[0]
            self.selected_collection_entry.config(state='normal')
            self.selected_collection_entry.delete(0, "end")
            self.selected_collection_entry.insert(0, collection_name)
            self.selected_collection_entry.config(state='readonly')

        if not collection_name:
            self.log_message("Error: No collection selected or derivable for restore.")
            return
        if not backup_file_or_path:
            self.log_message("Error: No backup file selected.")
            return

        self.log_message(f"Restore '{collection_name}' from '{backup_file_or_path}' mode '{restore_mode}'...")
        self.run_in_async_loop(self.restore_from_backup, collection_name, backup_file_or_path, restore_mode)

    # ---------- Backup / Restore Core ----------

    async def fetch_and_update_db_info(self):
        try:
            collections = await self.db.list_collection_names()
            db_stats = await self.db.command("dbstats")

            info_lines = [
                f"Database: {self.db_name}",
                f"Collections ({db_stats.get('collections', 0)}):",
                "--------------------",
                *sorted(collections),
                "\n--- DB Stats ---",
                f"Objects: {db_stats.get('objects', 'N/A')}",
                f"Data Size: {db_stats.get('dataSize', 0) / 1024 ** 2:.2f} MB",
                f"Storage Size: {db_stats.get('storageSize', 0) / 1024 ** 2:.2f} MB",
                f"Index Size: {db_stats.get('indexSize', 0) / 1024 ** 2:.2f} MB",
            ]
            info_str = "\n".join(info_lines)

            def _update_gui():
                self.db_info_text.delete('1.0', "end")
                self.db_info_text.insert("end", info_str)

            self.after(0, _update_gui)
        except Exception as e:
            logging.error(f"Failed to fetch DB info: {e}")
            self.log_message(f"Error fetching DB info: {e}")

    async def fetch_and_update_collections(self):
        try:
            names = await self.db.list_collection_names()

            def _update_gui():
                current_selection = self.collection_listbox.curselection()
                self.collection_listbox.delete(0, "end")
                for name in sorted(names):
                    self.collection_listbox.insert("end", name)
                if current_selection:
                    try:
                        self.collection_listbox.selection_set(current_selection)
                    except Exception:
                        pass  # Selection might be out of bounds after refresh

            self.after(0, _update_gui)
        except Exception as e:
            logging.error(f"Failed to fetch collections: {e}")

    def update_backup_files_listbox(self, collection_name: str):
        self.backup_files_listbox.delete(0, "end")
        try:
            pattern = re.compile(rf"^{re.escape(collection_name)}\[\d{{14}}\]\.(json|bson|csv)$", re.I)
            for file_path in self.backup_dir.iterdir():
                if file_path.is_file() and pattern.match(file_path.name):
                    self.backup_files_listbox.insert("end", file_path.name)
        except FileNotFoundError:
            self.log_message(f"Backup directory not found: {self.backup_dir}")
        except Exception as e:
            self.log_message(f"Error listing backup files: {e}")

    async def backup_collection(self, collection_name: str, fmt: str):
        try:
            collection = self.db[collection_name]
            cursor = collection.find({})
            docs = await cursor.to_list(length=None)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ext = fmt.lower()
            # This is where the error was, now it uses the Path object correctly.
            backup_file = self.backup_dir / f"{collection_name}[{timestamp}].{ext}"

            if fmt == "JSON":
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(json_util.dumps(docs, indent=2))
            elif fmt == "BSON":
                if not HAVE_BSON_STREAM:
                    self.log_message("BSON streaming not available; falling back to JSON.")
                    with open(backup_file.with_suffix(".json"), 'w', encoding='utf-8') as f:
                        f.write(json_util.dumps(docs, indent=2))
                else:
                    with open(backup_file, 'wb') as f:
                        for d in docs:
                            f.write(BSON.encode(d))
            elif fmt == "CSV":
                flat_rows, headers = [], set()
                for d in docs:
                    flat = self._flatten_for_csv(d)
                    flat_rows.append(flat)
                    headers.update(flat.keys())
                headers = sorted(list(headers))
                with open(backup_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for row in flat_rows:
                        safe_row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
                        writer.writerow(safe_row)
            else:
                raise ValueError(f"Unsupported backup format: {fmt}")

            self.log_message(f"Backed up {len(docs)} docs from '{collection_name}' to {backup_file.name}")
            self.after(0, lambda: self.update_backup_files_listbox(collection_name))
        except Exception as e:
            logging.error(f"Backup failed for '{collection_name}': {e}")
            self.log_message(f"Error during backup of '{collection_name}': {e}")

    async def backup_all_collections(self, fmt: str):
        try:
            collections = await self.db.list_collection_names()
            for name in collections:
                await self.backup_collection(name, fmt)
            self.log_message("Finished backing up all collections.")
        except Exception as e:
            logging.error(f"Backup all failed: {e}")
            self.log_message(f"Error during 'Backup All': {e}")

    async def restore_from_backup(self, collection_name: str, filename_or_path: str, mode: str):
        p = Path(filename_or_path)
        if not p.is_absolute():
            p = (self.backup_dir / filename_or_path).resolve()
        if not p.exists():
            self.log_message(f"Error: Backup file not found at {p}")
            return

        ext = p.suffix.lower()
        docs = []

        try:
            if ext == ".json":
                with open(p, 'r', encoding='utf-8') as f:
                    data = json_util.loads(f.read())
                docs = list(data) if isinstance(data, list) else [data]
            elif ext == ".bson":
                if not HAVE_BSON_STREAM:
                    self.log_message("BSON restore not supported in this environment.")
                    return
                with open(p, 'rb') as f:
                    docs = list(decode_file_iter(f))
            elif ext == ".csv":
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        docs.append(self._unflatten_from_csv(row))
            else:
                self.log_message(f"Unsupported restore file type: {ext}")
                return

            # normalize _id if stringified hex
            for d in docs:
                if isinstance(d, dict) and "_id" in d and isinstance(d["_id"], str) and ObjectId.is_valid(d["_id"]):
                    try:
                        d["_id"] = ObjectId(d["_id"])
                    except errors.InvalidId:
                        pass

            coll = self.db[collection_name]
            if mode.startswith("Replace"):
                await coll.delete_many({})
                if docs:
                    res = await coll.insert_many(docs)
                    self.log_message(f"Replace: inserted {len(res.inserted_ids)} docs into '{collection_name}'.")
            else:  # Merge (Upsert)
                if not docs:
                    self.log_message("Merge: no operations generated.")
                    return
                ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) if "_id" in d else InsertOne(d) for d in docs]
                result = await coll.bulk_write(ops, ordered=False)
                self.log_message(
                    f"Merge: inserted={result.inserted_count} matched={result.matched_count} modified={result.modified_count} upserted={result.upserted_count}")
        except BulkWriteError as bwe:
            self.log_message(f"Restore bulk write error: {bwe.details.get('nInserted', 0)} inserted. Check logs.")
            logging.error(f"BulkWriteError details: {bwe.details}")
        except Exception as e:
            logging.error(f"Restore failed: {e}")
            self.log_message(f"Restore error: {e}")

    # ---------- Collection Viewer Logic omitted for brevity ---
    # ... (the rest of the ZManager class remains the same)

    def cv_refresh_docs_clicked(self):
        self.cv_pager.skip = 0
        self.cv_refresh_docs(reset=True)

    def cv_load_more_clicked(self):
        self.cv_pager.skip += self.cv_pager.limit
        self.cv_refresh_docs(reset=False)

    def cv_refresh_docs(self, reset: bool):
        collection = (self.cv_collection_entry.get() or "").strip()
        if not collection:
            self.log_message("Collection is required (Collection Viewer).")
            return
        text = (self.cv_filter_entry.get() or "{}").strip()
        try:
            filt = json.loads(text)
            if not isinstance(filt, dict): raise ValueError
        except Exception:
            self.log_message("Invalid JSON filter; using {}.")
            filt = {}

        async def _fetch_ids():
            cursor = self.db[collection].find(filt, projection={"_id": 1}).skip(self.cv_pager.skip).limit(
                self.cv_pager.limit)
            items = await cursor.to_list(length=self.cv_pager.limit)
            return [(str(it.get("_id")), it.get("_id")) for it in items]

        def _update_ui(pairs):
            if reset:
                self.cv_doc_listbox.delete(0, END)
                self.cv_ids_cache = []
            self.cv_ids_cache.extend(pairs)
            for sid, _ in pairs:
                self.cv_doc_listbox.insert(END, sid)
            self.log_message(f"Loaded {len(pairs)} doc ids (total listed: {len(self.cv_ids_cache)}).")

        fut = self.run_in_async_loop(_fetch_ids)
        fut.add_done_callback(lambda f: self.after(0, lambda: _update_ui(f.result())))

    def cv_on_doc_select(self, event=None):
        idxs = self.cv_doc_listbox.curselection()
        if not idxs: return
        _id_str, _id_obj = self.cv_ids_cache[idxs[0]]
        collection = (self.cv_collection_entry.get() or "").strip()
        if not collection: return

        async def _fetch_doc():
            q = {"_id": _id_obj}
            if isinstance(_id_obj, str) and ObjectId.is_valid(_id_obj):
                q = {"_id": ObjectId(_id_obj)}
            return await self.zmongo.find_document(collection, q, cache=True)

        def _display(res: SafeResult):
            self.cv_json_text.delete("1.0", END)
            if not res.success or not res.data:
                self.cv_json_text.insert("end", f"Not found or error.\n{res.error or ''}")
                return
            try:
                pretty = json.dumps(res.data, indent=2, ensure_ascii=False)
            except Exception:
                pretty = json_util.dumps(res.data, indent=2)
            self.cv_json_text.insert("end", pretty)
            self.cv_id_entry.delete(0, END)
            self.cv_id_entry.insert(0, str(res.data.get("_id")))

        fut = self.run_in_async_loop(_fetch_doc)
        fut.add_done_callback(lambda f: self.after(0, lambda: _display(f.result())))

    def cv_insert_doc_dialog(self):
        def _submit():
            raw = txt.get("1.0", END)
            try:
                doc = json.loads(raw)
                if not isinstance(doc, dict): raise ValueError("JSON must be an object.")
            except Exception as e:
                info_label.config(text=f"Invalid JSON: {e}", foreground="red")
                return

            collection = (self.cv_collection_entry.get() or "").strip()
            if not collection:
                info_label.config(text="Collection required.", foreground="red")
                return

            async def _insert():
                return await self.zmongo.insert_document(collection, doc)

            def _done(f):
                res: SafeResult = f.result()
                if res.success:
                    info_label.config(text=f"Inserted: {res.data.get('inserted_id')}", foreground="green")
                    self.cv_refresh_docs_clicked()
                else:
                    info_label.config(text=f"Insert failed: {res.error}", foreground="red")

            fut = self.run_in_async_loop(_insert)
            fut.add_done_callback(lambda f: self.after(0, lambda: _done(f)))

        win = Toplevel(self)
        win.title("Insert Document (JSON)")
        win.geometry("700x500")
        txt = ScrolledText(win, font=("Courier New", 10), wrap="none")
        txt.pack(fill=BOTH, expand=True, padx=8, pady=8)
        txt.insert("end", "{\n  \n}")
        bottom = ttk.Frame(win)
        bottom.pack(fill=X, padx=8, pady=(0, 8))
        info_label = ttk.Label(bottom, text="Enter a JSON object and click Insert.")
        info_label.pack(side=LEFT)
        Button(bottom, text="Insert", command=_submit).pack(side=RIGHT)

    def cv_delete_selected(self):
        idxs = self.cv_doc_listbox.curselection()
        if not idxs:
            self.log_message("No document selected to delete.")
            return
        _id_str, _id_obj = self.cv_ids_cache[idxs[0]]
        collection = (self.cv_collection_entry.get() or "").strip()
        if not collection:
            self.log_message("Collection required for delete.")
            return

        async def _delete():
            q = {"_id": _id_obj}
            if isinstance(_id_obj, str) and ObjectId.is_valid(_id_obj):
                q = {"_id": ObjectId(_id_obj)}
            return await self.zmongo.delete_document(collection, q)

        def _done(f):
            res: SafeResult = f.result()
            if res.success and (res.data or {}).get("deleted_count", 0) >= 1:
                self.log_message(f"Deleted document: {_id_str}")
                self.cv_doc_listbox.delete(idxs[0])
                del self.cv_ids_cache[idxs[0]]
                self.cv_json_text.delete("1.0", END)
            else:
                self.log_message(f"Delete failed or not found: {res.error}")

        fut = self.run_in_async_loop(_delete)
        fut.add_done_callback(lambda f: self.after(0, lambda: _done(f)))

    def on_apply_dotkey_value_clicked(self):
        collection = (self.cv_collection_entry.get() or "").strip()
        dot_key = (self.cv_dotkey_entry.get() or "").strip()
        raw_value = self.cv_value_entry.get()
        raw_id = self.cv_id_entry.get()

        if not all([collection, dot_key, raw_id]):
            self.log_message("Error: Collection, Dot-key, and _id are required.")
            return

        q = {"_id": self._parse_objectid(raw_id) or raw_id}
        value = self._parse_input_value(raw_value)
        self.log_message(f"Applying $set on '{collection}' at '{dot_key}' for _id={q['_id']}...")

        async def _do_update():
            return await self.zmongo.update_document(collection, q, {"$set": {dot_key: value}})

        def _done(fut):
            res: SafeResult = fut.result()
            if res.success:
                meta = res.data or {}
                self.log_message(f"Success: matched={meta.get('matched_count')} modified={meta.get('modified_count')}.")
                self.cv_on_doc_select()
            else:
                self.log_message(f"Failed: {res.error}")

        self.run_in_async_loop(_do_update).add_done_callback(_done)

    def open_file_explorer(self):
        filepath = filedialog.askopenfilename(
            initialdir=str(self.backup_dir),
            title="Select a Backup File",
            filetypes=[("All supported", "*.json *.bson *.csv"), ("JSON files", "*.json"), ("BSON files", "*.bson"),
                       ("CSV files", "*.csv")]
        )
        if not filepath: return

        file_path_obj = Path(filepath)
        collection_name = file_path_obj.name.partition('[')[0]

        self.selected_backup_entry.config(state='normal')
        self.selected_backup_entry.delete(0, END)
        self.selected_backup_entry.insert(0, str(file_path_obj))
        self.selected_backup_entry.config(state='readonly')

        self.selected_collection_entry.config(state='normal')
        self.selected_collection_entry.delete(0, END)
        self.selected_collection_entry.insert(0, collection_name)
        self.selected_collection_entry.config(state='readonly')

        self.log_message(f"Selected file: {file_path_obj.name}")
        self.log_message(f"Inferred collection for restore: {collection_name}")

    def run_periodic_updates(self):
        self.run_in_async_loop(self.fetch_and_update_db_info)
        self.run_in_async_loop(self.fetch_and_update_collections)
        self.after(30000, self.run_periodic_updates)

    def on_closing(self):
        logging.info("Closing application and MongoDB connections.")
        self.async_client.close()
        self.sync_client.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()


def main():
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    app = ZManager(loop)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    loop_thread.join()


if __name__ == "__main__":
    main()
