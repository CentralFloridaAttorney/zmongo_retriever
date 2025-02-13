import json
import re
import subprocess
import threading
from datetime import datetime
from tkinter import ttk, filedialog

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import asyncio
from bson import ObjectId, errors
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
import tkinter as tk

from zmongo.zmongo_retriever import convert_json_to_metadata, get_value
load_dotenv()

# Use environment variables instead of zconstants
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DATABASE_NAME = os.getenv('MONGO_DB_NAME')
PROJECT_PATH = os.getenv('PROJECT_PATH')
MONGO_BACKUP_DIR = os.getenv('MONGO_BACKUP_DIR')

client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE_NAME]
chats_collection = db['chats']
users_collection = db['user']


# A helper function to run async tasks and update Tkinter from the main thread
def run_async_in_tkinter(async_func, loop, *args, **kwargs):
    def callback(future):
        try:
            result = future.result()
            if callable(result):
                app.after(0, result)
        except Exception as e:
            print("Async task error:", e)

    future = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), loop)
    future.add_done_callback(callback)


async def add_without_updating(collection, data):
    try:
        if not data:
            return "No data provided for insertion."

        operations = []
        for doc in data:
            # Convert 'creator' field to ObjectId if necessary
            if 'creator' in doc and isinstance(doc['creator'], str):
                doc['creator'] = ObjectId(doc['creator'])

            # Assign a new ObjectId to _id if not valid or not present
            try:
                if '_id' in doc:
                    doc['_id'] = ObjectId(str(doc['_id']))
                else:
                    doc['_id'] = ObjectId()
            except errors.InvalidId:
                doc['_id'] = ObjectId()

            # Append to operations if _id does not exist in the collection
            if not await collection.find_one({'_id': doc['_id']}):
                operations.append(InsertOne(doc))

        if not operations:
            return "No new document to insert."

        result = await collection.bulk_write(operations, ordered=False)
        return f"Inserted {result.inserted_count} document."

    except BulkWriteError as bwe:
        return f"Bulk write error: {bwe.details}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


async def add_and_update(db, collection_name, data):
    # Implementation
    pass


async def update_without_adding(db, collection_name, data):
    # Implementation
    pass


async def remove_all_and_replace(db, collection_name, data):
    # Implementation
    pass


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


async def backup_collection(mongo_uri, db_name, collection_name, backup_dir):
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Create backup directory if it doesn't exist
    make_dir_if_not_exists(backup_dir)

    # Filename: collection_name[TIMESTAMP].json
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_file = os.path.join(backup_dir, f"{collection_name}[{timestamp}].json")

    # Fetching all document from the collection
    documents = await collection.find({}).to_list(None)

    # Writing document to the file
    with open(backup_file, 'w') as file:
        json.dump(documents, file, default=str)  # Using str for non-serializable types

    print(f"Backup completed for collection {collection_name}. File: {backup_file}")


# GUI Class
class SystemManagerGUI(tk.Tk):
    def __init__(self, mongo_uri, db_name, project_dir, mongo_backup_dir, project_backup_dir):
        super().__init__()
        self.zmongo_retriever_process = None
        self.ocr_process = None
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.project_dir = project_dir
        self.mongo_backup_dir = mongo_backup_dir
        self.project_backup_dir = project_backup_dir
        self.extended_info_text = None
        self.basic_info_text = None
        self.selected_table_entry = None
        self.selected_backup_filename = None  # Class attribute to store selected filename
        self.table_listbox = None
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self._create_widgets()
        self.loop = asyncio.get_event_loop()
        self.after(100, self.run_async_tasks)
        self.backup_dir = os.path.join(self.project_dir, self.project_backup_dir, self.mongo_backup_dir)
        self.make_dir_if_not_exists(self.backup_dir)
        self.current_output_widgets = []

    def _create_widgets(self):
        # Create tabs for different functionalities
        tab_control = ttk.Notebook(self)
        basic_info_tab = ttk.Frame(tab_control)
        maintenance_tab = ttk.Frame(tab_control)
        collection_tab = ttk.Frame(tab_control)

        tab_control.add(basic_info_tab, text='Database Info')
        tab_control.add(maintenance_tab, text='Backup/Restore')
        tab_control.add(collection_tab, text='Collection')

        self.notebook = ttk.Notebook(self)
        self.start_system_tab = ttk.Frame(self.notebook)
        self.output_tab_flask = ttk.Frame(self.notebook)
        self.output_tab_llm = ttk.Frame(self.notebook)
        self.output_tab_ocr = ttk.Frame(self.notebook)
        self.output_tab_zmongo_retriever = ttk.Frame(self.notebook)

        self.notebook.add(self.start_system_tab, text='Start System')
        self.notebook.add(self.output_tab_flask, text='Console Outputs')

        self.notebook.pack(expand=1, fill="both")

        # Buttons in Start System Tab
        tk.Button(self.start_system_tab, text="Start ZMongoRetriever", command=self.start_zmongo_retriever).pack(
            pady=10)

        # Output Area for ZMongoRetriever App
        self.zmongo_retriever_output_text = tk.Text(self.output_tab_flask)
        self.zmongo_retriever_output_text.pack(expand=True, fill='both')

        # Widgets for Basic Info Tab
        self.basic_info_text = tk.Text(basic_info_tab)
        self.basic_info_text.pack(expand=True, fill='both')

        # Listbox for tables
        self.table_listbox = tk.Listbox(maintenance_tab, exportselection=False)
        self.table_listbox.bind('<<ListboxSelect>>', self.on_table_select)
        self.table_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.selected_table_entry = tk.Entry(maintenance_tab, state='readonly')
        self.selected_table_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Listbox for backup files
        self.backup_files_listbox = tk.Listbox(maintenance_tab, exportselection=False)
        self.backup_files_listbox.bind('<<ListboxSelect>>', self.on_backup_file_select)
        self.backup_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Dropdown for restore options
        self.restore_options = ttk.Combobox(maintenance_tab, state='readonly', values=[
            "Add without Updating",
            "Add and Update",
            "Update without Adding",
            "Remove All & Replace"
        ])
        self.restore_options.current(0)  # Set default value
        self.restore_options.pack()

        # Entry for displaying the selected backup file
        self.selected_backup_file_entry = tk.Entry(maintenance_tab, state='readonly')
        self.selected_backup_file_entry.bind("<FocusIn>", self.on_entry_click)
        self.selected_backup_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        backup_button = tk.Button(maintenance_tab, text='Backup', command=self.on_backup_button_clicked)
        restore_button = tk.Button(maintenance_tab, text='Restore', command=self.on_restore_button_clicked)
        backup_all_button = tk.Button(maintenance_tab, text='Backup All', command=self.on_backup_all_button_clicked)
        find_backups_button = tk.Button(maintenance_tab, text='Find Backups', command=self.find_and_list_backups)
        open_file_button = tk.Button(maintenance_tab, text='Open File', command=self.open_file_explorer)
        open_file_button.pack()
        find_backups_button.pack()
        backup_all_button.pack()
        backup_button.pack()
        restore_button.pack()

        self.message_text = tk.Text(maintenance_tab, height=4, state='disabled')
        self.message_text.pack(fill=tk.BOTH, expand=True)

        # Setup for Collection Tab
        self.tree = ttk.Treeview(collection_tab, columns=("Column1", "Column2"), show="headings")
        self.tree.heading('Column1', text='Column 1')
        self.tree.heading('Column2', text='Column 2')

        # Specify column configuration
        min_width = 16 * 8
        self.tree.column('Column1', width=min_width, minwidth=min_width, stretch=tk.YES)
        self.tree.column('Column2', width=min_width, minwidth=min_width, stretch=tk.YES)

        # Create scrollbars
        vscroll = ttk.Scrollbar(collection_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscroll.set)
        hscroll = ttk.Scrollbar(collection_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hscroll.set)

        # Layout the treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew', in_=collection_tab)
        vscroll.grid(row=0, column=1, sticky='ns', in_=collection_tab)
        hscroll.grid(row=1, column=0, sticky='ew', in_=collection_tab)

        collection_tab.grid_columnconfigure(0, weight=1)
        collection_tab.grid_rowconfigure(0, weight=1)

        # Button to load the selected collection
        load_collection_button = tk.Button(collection_tab, text="Load Collection", command=self.load_collection_data)
        load_collection_button.grid(row=2, column=0, columnspan=2, pady=10, sticky='ew')

        tab_control.pack(expand=1, fill="both")

    def append_message(self, message):
        self.message_text.config(state='normal')
        self.message_text.insert(tk.END, message + "\n")
        self.message_text.config(state='disabled')

    def async_load_collection_data(self, collection_name):
        def target():
            self.fetch_and_display(collection_name)

        threading.Thread(target=target, daemon=True).start()

    async def backup_all_collections(self):
        collections = await self.db.list_collection_names()
        for collection_name in collections:
            await backup_collection(os.getenv("MONGO_URI"), os.getenv("MONGO_DATABASE_NAME"), collection_name,
                                    self.backup_dir)
            print(f"Backup completed for collection: {collection_name}")

    async def backup_collection(self, mongo_uri, db_name, collection_name, backup_dir):
        """
        Asynchronously backs up a specified MongoDB collection to a JSON file.
        """
        client = AsyncIOMotorClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        self.make_dir_if_not_exists(backup_dir)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(backup_dir, f"{collection_name}_{timestamp}.json")

        documents = await collection.find({}).to_list(length=None)

        with open(backup_file, 'w') as file:
            json.dump(documents, file, default=str)

        message = f"Backup of collection '{collection_name}' completed. File: {backup_file}"
        print(message)
        client.close()

    async def backup_selected_collection(self, selected_collection):
        await backup_collection(os.getenv("MONGO_URI"), os.getenv("MONGO_DATABASE_NAME"), selected_collection,
                                self.backup_dir)

    def fetch_and_display(self, collection_name):
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        mongo_db = mongo_client[os.getenv("MONGO_DATABASE_NAME")]
        collection = mongo_db[collection_name]
        records = collection.find({})
        self.setup_and_populate_treeview(records)

    async def fetch_basic_info(self):
        try:
            collections = await self.db.list_collection_names()
            db_stats = await self.db.command("dbstats")

            info_str = f"Database: {self.db_name}\n"
            info_str += "Collections:\n" + "\n".join(["- " + col for col in collections]) + "\n"
            info_str += "Database Stats:\n"
            info_str += "\n".join([f"{k}: {v}" for k, v in db_stats.items()])

            return lambda: self.update_basic_info_gui(info_str)
        except Exception as e:
            return lambda: self.update_basic_info_gui(f"Error fetching basic info: {str(e)}")

    def get_selected_collection(self):
        try:
            selected_index = self.table_listbox.curselection()
            if selected_index:
                selected_collection = self.table_listbox.get(selected_index)
                return selected_collection
            else:
                print("No collection selected")
                return None
        except Exception as e:
            print("Error in getting selected collection:", e)
            return None

    async def fetch_extended_info(self):
        pass

    def find_and_list_backups(self):
        try:
            all_backup_files = os.listdir(self.backup_dir)
            filtered_files = [file for file in all_backup_files if re.match(r'.*\[\d{14}]\.json', file)]
            self.backup_files_listbox.delete(0, tk.END)
            for file in filtered_files:
                self.backup_files_listbox.insert(tk.END, file)
        except Exception as e:
            print("Error listing backup files:", e)

    def load_collection_data(self):
        collection_name = self.selected_table_entry.get()
        if collection_name:
            self.tree.delete(*self.tree.get_children())
            mongo_client = MongoClient(os.getenv("MONGO_URI"))
            mongo_db = mongo_client[os.getenv("MONGO_DB_NAME")]
            collection = mongo_db[collection_name]
            records = collection.find({})
            self.setup_and_populate_treeview(records)
        else:
            print("No collection selected.")

    @staticmethod
    def make_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_backup_all_button_clicked(self):
        if not self.loop.is_running():
            self.loop.run_until_complete(self.backup_all_collections())
        else:
            asyncio.run_coroutine_threadsafe(self.backup_all_collections(), self.loop)
            message = f"Backup of all tables."
            self.append_message(message)

    def on_backup_button_clicked(self):
        selected_collection = self.get_selected_collection()
        if selected_collection:
            if not self.loop.is_running():
                self.loop.run_until_complete(self.backup_selected_collection(selected_collection))
                self.append_message(f"Backup complete for table: {selected_collection}")
            else:
                asyncio.run_coroutine_threadsafe(self.backup_selected_collection(selected_collection), self.loop)
                message = f"Backup of collection '{selected_collection}'."
                self.append_message(message)
        else:
            print("No collection selected for backup")

    def on_backup_file_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.selected_backup_filename = event.widget.get(index)

            self.selected_backup_file_entry.config(state='normal')
            self.selected_backup_file_entry.delete(0, tk.END)
            self.selected_backup_file_entry.insert(0, self.selected_backup_filename)
            self.selected_backup_file_entry.config(state='readonly')

            self.append_message(f"Selected backup file: {self.selected_backup_filename}")

    @staticmethod
    def on_entry_click(event):
        event.widget.config(state='normal')
        event.widget.select_range(0, tk.END)
        event.widget.config(state='readonly')

    def on_restore_button_clicked(self):
        if not self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.restore_db(), self.loop)
        else:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.restore_db()))

    def on_table_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            table_name = event.widget.get(index)
            self.selected_table_entry.config(state=tk.NORMAL)
            self.selected_table_entry.delete(0, tk.END)
            self.selected_table_entry.insert(0, table_name)
            self.selected_table_entry.config(state='readonly')
        self.update_backup_files_list()

    def open_file_explorer(self):
        file_path = filedialog.askopenfilename(initialdir=self.backup_dir, filetypes=[("JSON files", "*.json")])
        if file_path:
            file_name = os.path.basename(file_path)
            self.selected_backup_file_entry.config(state='normal')
            self.selected_backup_file_entry.delete(0, tk.END)
            self.selected_backup_file_entry.insert(0, file_name)
            self.selected_backup_file_entry.config(state='readonly')
            self.selected_backup_filename = file_name

    def read_backup_file(self, file_name):
        file_path = os.path.join(self.backup_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            self.append_message(f"Error reading file: {e}")
            return None

    async def restore_db(self):
        selected_collection = self.get_selected_collection()
        selected_option = self.restore_options.get()

        if not self.selected_backup_filename:
            self.append_message("No file selected for restore")
            return

        data = self.read_backup_file(self.selected_backup_filename)
        if data is None:
            return

        data = self.preprocess_data(data)

        collection_name = (selected_collection if selected_collection
                           else self.selected_backup_filename.partition('[')[0])
        collection_to_restore = self.db[collection_name]

        try:
            if selected_option == "Add without Updating":
                inserted_count = await add_without_updating(collection_to_restore, data)
                self.append_message(
                    f"Restored {inserted_count} records to {collection_name} using 'Add without Updating'")
            # Add your other restore operations here...
        except Exception as e:
            self.append_message(f"Restore operation error: {str(e)}")

    @staticmethod
    def preprocess_data(data):
        for doc in data:
            if 'user_id' in doc:
                doc['creator'] = doc.pop('user_id')
            if 'created_by' in doc:
                doc['creator'] = doc.pop('created_by')
            if 'created_at' in doc:
                doc['date_time'] = doc.pop('created_at')
            if 'creator' in doc and isinstance(doc['creator'], str):
                user = users_collection.find_one({'user_id': doc['creator']})
                doc['creator'] = user['_id'] if user else ObjectId()
        return data

    def run_async_tasks(self):
        run_async_in_tkinter(self.fetch_basic_info, self.loop)
        run_async_in_tkinter(self.update_table_list, self.loop)
        self.update_backup_files_list()
        self.after(10000, self.run_async_tasks)

    def run_zmongo_retriever(self):
        self.zmongo_retriever_process = self.run_program(
            "zmongo_retriever.py",
            self.zmongo_retriever_output_text
        )

    def run_program(self, program_path, output_widget=None):
        if output_widget not in self.current_output_widgets:
            self.current_output_widgets.append(output_widget)

            def target():
                process = subprocess.Popen(['python', program_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           text=True)
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    if output_widget:
                        self.update_output_text(output_widget, line)
                    else:
                        print(line, end='')
                process.wait()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            return thread

    def setup_and_populate_treeview(self, records):
        self.tree.delete(*self.tree.get_children())

        records_list = list(records)
        if not records_list:
            return

        column_names = list(convert_json_to_metadata(records_list[0]).keys())
        self.tree["columns"] = column_names

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col, anchor='w')
            self.tree.column(col, anchor="w", width=120)

        for record in records_list:
            values = [get_value(record, col) for col in column_names]
            self.tree.insert("", "end", values=values)

    def start_zmongo_retriever(self):
        threading.Thread(target=self.run_zmongo_retriever, daemon=True).start()

    def update_backup_files_list(self):
        selected_collection = self.get_selected_collection()
        if not selected_collection:
            return
        try:
            backup_dir_path = os.path.exists(self.backup_dir)
            all_backup_files = os.listdir(backup_dir_path)
            filtered_files = []
            for file in all_backup_files:
                file_table_name, _, _ = file.partition('[')
                if file_table_name == selected_collection:
                    filtered_files.append(file)

            self.backup_files_listbox.delete(0, tk.END)
            for file in filtered_files:
                self.backup_files_listbox.insert(tk.END, file)
        except Exception as e:
            print("Error listing backup files:", e)

    def update_basic_info_gui(self, info_str):
        self.basic_info_text.delete('1.0', tk.END)
        self.basic_info_text.insert(tk.END, info_str)

    def update_table_list_gui(self, collections):
        self.table_listbox.delete(0, tk.END)
        for collection in collections:
            self.table_listbox.insert(tk.END, collection)

    @staticmethod
    def update_output_text(output_widget, message):
        output_widget.insert(tk.END, message)
        output_widget.see(tk.END)

    async def update_table_list(self):
        try:
            collections = await self.db.list_collection_names()
            self.table_listbox.delete(0, tk.END)
            for collection in collections:
                self.table_listbox.insert(tk.END, collection)
        except Exception as e:
            print("Error updating table list:", e)


if __name__ == "__main__":
    app = SystemManagerGUI(
        mongo_uri=os.getenv("MONGO_URI"),
        db_name=os.getenv("MONGO_DB_NAME"),
        project_dir=os.getenv("PROJECT_PATH"),
        mongo_backup_dir=os.getenv("PROJECT_BACKUP_DIR"),# Or whichever directory logic you prefer
        project_backup_dir="xyzzy"
    )

    # Start the asyncio event loop in a separate thread
    loop_thread = threading.Thread(target=app.loop.run_forever, daemon=True)
    loop_thread.start()

    app.mainloop()
