import asyncio
import json
import logging
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, Text, Tk, Listbox, Entry, Button, Frame

from bson import errors
from bson.objectid import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, InsertOne, UpdateOne, DeleteOne, ReplaceOne
from pymongo.errors import BulkWriteError

# --- Configuration and Setup ---
# It's better to define a base directory for the application
load_dotenv(Path.home() / "resources" / ".env_local")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables with sensible defaults
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "default_db")
# Define a default backup directory relative to the app's location
MONGO_BACKUP_DIR = Path(os.getenv("MONGO_BACKUP_DIR", './tmp'))


class ZMongoSystemManager(Tk):
    """
    A Tkinter GUI application for managing a ZMongo database, including backups,
    restores, and running associated services.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.title("ZMongo System Manager")
        self.geometry("1200x800")

        self.loop = loop
        self.db_name = MONGO_DATABASE_NAME
        # Use pathlib for robust path management
        self.backup_dir = MONGO_BACKUP_DIR / self.db_name
        self.make_dir_if_not_exists(self.backup_dir)

        # MongoDB clients
        try:
            self.async_client = AsyncIOMotorClient(MONGO_URI)
            self.db = self.async_client[self.db_name]
            self.sync_client = MongoClient(MONGO_URI)
            self.sync_db = self.sync_client[self.db_name]
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            self.destroy()
            return

        self._create_widgets()
        self.run_periodic_updates()

    def _create_widgets(self):
        """Create and layout all the widgets for the GUI."""
        main_notebook = ttk.Notebook(self)
        main_notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Tabs ---
        db_info_tab = ttk.Frame(main_notebook)
        maintenance_tab = ttk.Frame(main_notebook)
        collection_tab = ttk.Frame(main_notebook)
        system_tab = ttk.Frame(main_notebook)

        main_notebook.add(db_info_tab, text='Database Info')
        main_notebook.add(maintenance_tab, text='Backup & Restore')
        main_notebook.add(collection_tab, text='Collection Viewer')
        main_notebook.add(system_tab, text='System Runner')

        # --- Database Info Tab ---
        self.db_info_text = Text(db_info_tab, wrap="word", font=("Courier New", 10))
        self.db_info_text.pack(expand=True, fill="both", padx=5, pady=5)

        # --- Maintenance Tab ---
        maint_frame = Frame(maintenance_tab)
        maint_frame.pack(fill="both", expand=True, padx=5, pady=5)
        maint_frame.grid_columnconfigure(1, weight=1)
        maint_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(maint_frame, text="Collections:").grid(row=0, column=0, sticky="w", padx=5)
        self.collection_listbox = Listbox(maint_frame, exportselection=False, height=10)
        self.collection_listbox.grid(row=1, column=0, rowspan=4, sticky="nswe", padx=5)
        self.collection_listbox.bind('<<ListboxSelect>>', self.on_collection_select)

        ttk.Label(maint_frame, text="Selected Collection:").grid(row=0, column=1, sticky="w", padx=5)
        self.selected_collection_entry = Entry(maint_frame, state='readonly')
        self.selected_collection_entry.grid(row=1, column=1, sticky="we", padx=5)

        ttk.Label(maint_frame, text="Backup Files:").grid(row=0, column=2, sticky="w", padx=5)
        self.backup_files_listbox = Listbox(maint_frame, exportselection=False, height=10)
        self.backup_files_listbox.grid(row=1, column=2, rowspan=4, sticky="nswe", padx=5)
        self.backup_files_listbox.bind('<<ListboxSelect>>', self.on_backup_file_select)

        ttk.Label(maint_frame, text="Selected Backup File:").grid(row=0, column=3, sticky="w", padx=5)
        self.selected_backup_entry = Entry(maint_frame, state='readonly')
        self.selected_backup_entry.grid(row=1, column=3, sticky="we", padx=5)

        # Action Buttons
        button_frame = ttk.Frame(maint_frame)
        button_frame.grid(row=2, column=1, columnspan=3, sticky="we", pady=10)
        Button(button_frame, text='Backup Selected', command=self.on_backup_selected_clicked).pack(side="left", padx=5)
        Button(button_frame, text='Backup All', command=self.on_backup_all_clicked).pack(side="left", padx=5)
        Button(button_frame, text='Restore Selected', command=self.on_restore_clicked).pack(side="left", padx=5)
        Button(button_frame, text='Browse for File...', command=self.open_file_explorer).pack(side="left", padx=5)

        self.restore_options = ttk.Combobox(maint_frame, state='readonly', values=[
            "Add without Updating", "Add and Update", "Update without Adding", "Remove All & Replace"
        ])
        self.restore_options.current(0)
        self.restore_options.grid(row=3, column=1, columnspan=3, sticky="we", padx=5)

        # Message Log
        self.message_text = Text(maint_frame, height=8, state='disabled', wrap="word")
        self.message_text.grid(row=5, column=0, columnspan=4, sticky="nswe", padx=5, pady=5)
        maint_frame.grid_rowconfigure(5, weight=1)

    def run_in_async_loop(self, async_func, *args, **kwargs):
        """Safely run an async function from the Tkinter thread."""
        future = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), self.loop)
        future.add_done_callback(self.on_async_task_done)

    def on_async_task_done(self, future):
        """Handle exceptions from async tasks."""
        try:
            future.result()
        except Exception as e:
            logging.error(f"Async task failed: {e}")
            self.log_message(f"Error: {e}")

    def log_message(self, message):
        """Append a message to the message log widget."""

        def _append():
            self.message_text.config(state='normal')
            self.message_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
            self.message_text.config(state='disabled')
            self.message_text.see("end")

        self.after(0, _append)

    # --- GUI Event Handlers ---

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
        collection_name = self.selected_collection_entry.get()
        if not collection_name:
            self.log_message("Error: No collection selected for backup.")
            return
        self.log_message(f"Starting backup for '{collection_name}'...")
        self.run_in_async_loop(self.backup_collection, collection_name)

    def on_backup_all_clicked(self):
        self.log_message("Starting backup for all collections...")
        self.run_in_async_loop(self.backup_all_collections)

    def on_restore_clicked(self):
        collection_name = self.selected_collection_entry.get()
        backup_file_or_path = self.selected_backup_entry.get()
        restore_mode = self.restore_options.get()

        # If collection name is not selected in the UI, try to derive it from the backup filename
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

        self.log_message(
            f"Starting restore for '{collection_name}' from '{backup_file_or_path}' using mode '{restore_mode}'...")
        self.run_in_async_loop(self.restore_from_backup, collection_name, backup_file_or_path, restore_mode)

    # --- Core Logic ---

    async def fetch_and_update_db_info(self):
        """Fetches DB stats and updates the GUI."""
        try:
            collections = await self.db.list_collection_names()
            db_stats = await self.db.command("dbstats")

            info_lines = [
                f"Database: {self.db_name}",
                f"Collections ({db_stats.get('collections', 0)}):",
                "--------------------",
                *collections,
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
        """Fetches collection names and updates the listbox."""
        try:
            collections = await self.db.list_collection_names()

            def _update_gui():
                current_selection = self.collection_listbox.curselection()
                self.collection_listbox.delete(0, "end")
                for name in sorted(collections):
                    self.collection_listbox.insert("end", name)
                if current_selection:
                    self.collection_listbox.selection_set(current_selection)

            self.after(0, _update_gui)
        except Exception as e:
            logging.error(f"Failed to fetch collections: {e}")

    def update_backup_files_listbox(self, collection_name: str):
        """Updates the backup files listbox for the selected collection."""
        self.backup_files_listbox.delete(0, "end")
        try:
            # Regex to match files like 'collection_name[timestamp].json'
            pattern = re.compile(rf"^{re.escape(collection_name)}\[\d{{14}}\]\.json$")
            for file_path in self.backup_dir.iterdir():
                if file_path.is_file() and pattern.match(file_path.name):
                    self.backup_files_listbox.insert("end", file_path.name)
        except FileNotFoundError:
            self.log_message(f"Backup directory not found: {self.backup_dir}")
        except Exception as e:
            self.log_message(f"Error listing backup files: {e}")

    async def backup_collection(self, collection_name: str):
        """Backs up a single collection to a JSON file."""
        try:
            collection = self.db[collection_name]
            documents = await collection.find({}).to_list(length=None)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # Use pathlib for safe path construction
            backup_file = self.backup_dir / f"{collection_name}[{timestamp}].json"

            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, default=str, indent=2)

            self.log_message(f"Successfully backed up {len(documents)} documents from '{collection_name}'.")
            self.after(0, lambda: self.update_backup_files_listbox(collection_name))
        except Exception as e:
            logging.error(f"Backup failed for '{collection_name}': {e}")
            self.log_message(f"Error during backup of '{collection_name}': {e}")

    async def backup_all_collections(self):
        """Backs up all collections in the database."""
        try:
            collections = await self.db.list_collection_names()
            for name in collections:
                await self.backup_collection(name)
            self.log_message("Finished backing up all collections.")
        except Exception as e:
            logging.error(f"Backup all failed: {e}")
            self.log_message(f"Error during 'Backup All': {e}")

    async def restore_from_backup(self, collection_name: str, filename_or_path: str, mode: str):
        """Restores a collection from a backup file based on the selected mode."""
        backup_file_path = Path(filename_or_path)
        # If the path is not absolute, it's a relative filename from our default backup dir
        if not backup_file_path.is_absolute():
            backup_file_path = self.backup_dir / filename_or_path

        if not backup_file_path.exists():
            self.log_message(f"Error: Backup file not found at {backup_file_path}")
            return

        try:
            with open(backup_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.log_message(f"Read {len(data)} documents from backup file.")
            collection = self.db[collection_name]

            # Pre-process data (handle ObjectIds)
            for doc in data:
                if '_id' in doc and isinstance(doc['_id'], dict) and '$oid' in doc['_id']:
                    doc['_id'] = ObjectId(doc['_id']['$oid'])
                elif '_id' in doc and isinstance(doc['_id'], str):
                    try:
                        doc['_id'] = ObjectId(doc['_id'])
                    except errors.InvalidId:
                        self.log_message(f"Warning: Invalid _id '{doc['_id']}' found, a new one will be generated.")
                        del doc['_id']

            # --- Restore Logic ---
            if mode == "Add without Updating":
                ops = [InsertOne(doc) for doc in data]
                if not ops:
                    self.log_message("No new documents to insert.")
                    return
                # Use ordered=False to continue on duplicate key errors
                result = await collection.bulk_write(ops, ordered=False)
                self.log_message(f"Restore complete. Inserted: {result.inserted_count} documents.")

            elif mode == "Remove All & Replace":
                await collection.delete_many({})
                if data:
                    result = await collection.insert_many(data)
                    self.log_message(f"Collection cleared. Restored {len(result.inserted_ids)} documents.")
                else:
                    self.log_message("Collection cleared. No documents to restore.")

            # TODO: Implement "Add and Update" and "Update without Adding"
            else:
                self.log_message(f"Restore mode '{mode}' is not yet implemented.")

        except json.JSONDecodeError:
            self.log_message(f"Error: Could not decode JSON from {filename_or_path}.")
        except BulkWriteError as bwe:
            self.log_message(
                f"Restore bulk write error: {bwe.details.get('nInserted', 0)} inserted. Check logs for details.")
            logging.error(f"BulkWriteError details: {bwe.details}")
        except Exception as e:
            logging.error(f"Restore failed: {e}")
            self.log_message(f"An unexpected error occurred during restore: {e}")

    def open_file_explorer(self):
        """Opens a file dialog to select a backup file manually."""
        filepath = filedialog.askopenfilename(
            initialdir=self.backup_dir,
            title="Select a Backup File",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        file_path_obj = Path(filepath)
        collection_name = file_path_obj.name.partition('[')[0]

        # Set the selected backup file in the GUI using its full path
        self.selected_backup_entry.config(state='normal')
        self.selected_backup_entry.delete(0, "end")
        self.selected_backup_entry.insert(0, str(file_path_obj))
        self.selected_backup_entry.config(state='readonly')

        # Directly set the collection entry as well.
        self.selected_collection_entry.config(state='normal')
        self.selected_collection_entry.delete(0, "end")
        self.selected_collection_entry.insert(0, collection_name)
        self.selected_collection_entry.config(state='readonly')

        self.log_message(f"Manually selected file: {file_path_obj.name}")
        self.log_message(f"Inferred collection for restore: {collection_name}")

    def run_periodic_updates(self):
        """Periodically fetches new data to keep the GUI up-to-date."""
        self.run_in_async_loop(self.fetch_and_update_db_info)
        self.run_in_async_loop(self.fetch_and_update_collections)
        self.after(30000, self.run_periodic_updates)  # Update every 30 seconds

    @staticmethod
    def make_dir_if_not_exists(directory: Path):
        """Creates a directory if it doesn't exist."""
        directory.mkdir(parents=True, exist_ok=True)

    def on_closing(self):
        """Handle window closing event."""
        logging.info("Closing application and MongoDB connections.")
        self.async_client.close()
        self.sync_client.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()


def main():
    """Main function to set up and run the application."""
    # Create a separate thread for the asyncio event loop
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    app = ZMongoSystemManager(loop)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    # The loop should be stopped when the app closes
    loop_thread.join()


if __name__ == "__main__":
    main()
