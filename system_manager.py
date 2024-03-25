import json
import re
import subprocess
import threading
from datetime import datetime
from tkinter import ttk, filedialog
from motor.motor_asyncio import AsyncIOMotorClient
import os
import asyncio
from bson import ObjectId, errors
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
import tkinter as tk

import zconstants
from zmongo_retriever import convert_json_to_metadata, get_value

client = MongoClient(zconstants.MONGO_URI)
db = client[zconstants.MONGO_DATABASE_NAME]
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
        # # Get the directory of the current Python script
        # current_script_dir = os.path.dirname(os.path.abspath(__file__))
        #
        # # Navigate up one directory to get the project directory
        # project_path = os.path.join(current_script_dir, base_dir)
        #
        # print("Project path:", project_path)
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
        # tk.Button(self.start_system_tab, text="Start OCR Runner", command=self.start_ocr_runner).pack(pady=10)
        # tk.Button(self.start_system_tab, text="Start Flask App", command=self.start_flask_app).pack(pady=10)
        # tk.Button(self.start_system_tab, text="Start LLM Runner", command=self.start_llm_runner).pack(pady=10)
        tk.Button(self.start_system_tab, text="Start ZMongoRetriever", command=self.start_zmongo_retriever).pack(
            pady=10)

        # # Output Area for Flask App
        # self.flask_output_text = tk.Text(self.output_tab_flask)
        # self.flask_output_text.pack(expand=True, fill='both')
        # # Output Area for LLM App
        # self.llm_output_text = tk.Text(self.output_tab_flask)
        # self.llm_output_text.pack(expand=True, fill='both')
        # # Output Area for OCR App
        # self.ocr_output_text = tk.Text(self.output_tab_flask)
        # self.ocr_output_text.pack(expand=True, fill='both')
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
        # Define the tree view with an initial set of columns
        self.tree = ttk.Treeview(collection_tab, columns=("Column1", "Column2"), show="headings")
        self.tree.heading('Column1', text='Column 1')
        self.tree.heading('Column2', text='Column 2')

        # Specify column configuration to make them resizable and set a minimum width
        min_width = 16 * 8  # Assuming roughly 8 pixels per character
        self.tree.column('Column1', width=min_width, minwidth=min_width, stretch=tk.YES)
        self.tree.column('Column2', width=min_width, minwidth=min_width, stretch=tk.YES)

        # Scrollbar setup code remains the same...
        # Create a vertical scrollbar
        vscroll = ttk.Scrollbar(collection_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscroll.set)

        # Create a horizontal scrollbar
        hscroll = ttk.Scrollbar(collection_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hscroll.set)

        # Layout the treeview and scrollbars in the collection tab using grid
        self.tree.grid(row=0, column=0, sticky='nsew', in_=collection_tab)
        vscroll.grid(row=0, column=1, sticky='ns', in_=collection_tab)
        hscroll.grid(row=1, column=0, sticky='ew', in_=collection_tab)

        # Configure grid to be responsive
        collection_tab.grid_columnconfigure(0, weight=1)
        collection_tab.grid_rowconfigure(0, weight=1)

        # Button to load the selected collection
        load_collection_button = tk.Button(collection_tab, text="Load Collection", command=self.load_collection_data)
        load_collection_button.grid(row=2, column=0, columnspan=2, pady=10, sticky='ew')

        # Your existing setup continues...
        tab_control.pack(expand=1, fill="both")

    def append_message(self, message):
        self.message_text.config(state='normal')
        self.message_text.insert(tk.END, message + "\n")
        self.message_text.config(state='disabled')

    def async_load_collection_data(self, collection_name):
        def target():
            self.fetch_and_display(collection_name)

        # Launch the thread
        threading.Thread(target=target, daemon=True).start()

    async def backup_all_collections(self):
        collections = await self.db.list_collection_names()
        for collection_name in collections:
            await backup_collection(zconstants.MONGO_URI, zconstants.MONGO_DATABASE_NAME, collection_name,
                                    self.backup_dir)
            print(f"Backup completed for collection: {collection_name}")

    async def backup_collection(self, mongo_uri, db_name, collection_name, backup_dir):
        """
        Asynchronously backs up a specified MongoDB collection to a JSON file.

        This function connects to the MongoDB database using the provided MongoDB URI,
        accesses the specified collection, and writes its contents to a JSON file.
        The JSON file is named with the collection name and a timestamp, and is saved
        in the specified backup directory.

        Parameters:
        - mongo_uri (str): MongoDB URI string for connecting to the database.
        - db_name (str): The name of the database containing the collection.
        - collection_name (str): The name of the collection to be backed up.
        - backup_dir (str): The directory path where the backup file will be stored.

        Example usage:
        ```
        backup = DatabaseBackup()
        await backup.backup_collection(
            mongo_uri="mongodb://localhost:27017",
            db_name="myDatabase",
            collection_name="myCollection",
            backup_dir="/path/to/backup/directory"
        )
        ```

        The backup file will be named in the format 'collectionName[YYYYMMDDHHMMSS].json',
        where 'YYYYMMDDHHMMSS' is the current timestamp.

        Note:
        - The directory specified in `backup_dir` must be writable.
        - The function prints a message upon successful completion of the backup.
        - This function requires an asyncio environment to run properly.
        """

        # Create a MongoDB client
        client = AsyncIOMotorClient(mongo_uri)

        # Access the specified database and collection
        db = client[db_name]
        collection = db[collection_name]

        # Create the backup directory if it doesn't exist
        self.make_dir_if_not_exists(backup_dir)

        # Prepare the filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(backup_dir, f"{collection_name}_{timestamp}.json")

        # Fetch all document from the collection
        documents = await collection.find({}).to_list(length=None)

        # Write the document to the backup file
        with open(backup_file, 'w') as file:
            json.dump(documents, file, default=str)

        # Confirmation message
        message = f"Backup of collection '{collection_name}' completed. File: {backup_file}"
        print(message)

        # Close the MongoDB client connection
        client.close()

    async def backup_selected_collection(self, selected_collection):
        await backup_collection(zconstants.MONGO_URI, zconstants.MONGO_DATABASE_NAME, selected_collection,
                                self.backup_dir)

    def fetch_and_display(self, collection_name):
        mongo_client = MongoClient(zconstants.MONGO_URI)
        mongo_db = mongo_client[zconstants.MONGO_DATABASE_NAME]
        collection = mongo_db[collection_name]
        records = collection.find({})
        # records = [doc async for doc in cursor]  # Use asynchronous list comprehension to fetch documents

        # Since the setup_and_populate_treeview function is likely synchronous and you're updating a GUI,
        # you'll need to ensure that this part of the code runs on the main thread or in a synchronous context.
        # If you're using a framework like tkinter, you might need to schedule the GUI update on the main thread.
        # This example does not cover that part since it heavily depends on the GUI framework you're using.

        # Assuming setup_and_populate_treeview can be called with await or run in an async context:
        self.setup_and_populate_treeview(records)

    async def fetch_basic_info(self):
        try:
            collections = await self.db.list_collection_names()
            db_stats = await self.db.command("dbstats")

            info_str = "Database: {}\n".format(self.db_name)
            info_str += "Collections:\n" + "\n".join(["- " + col for col in collections]) + "\n"
            info_str += "Database Stats:\n"
            info_str += "\n".join(["{}: {}".format(k, v) for k, v in db_stats.items()])

            # Return a function to update the GUI
            return lambda: self.update_basic_info_gui(info_str)
        except Exception as e:
            return lambda: self.update_basic_info_gui("Error fetching basic info: {}".format(str(e)))

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
        # Asynchronously fetch extended information about the database
        pass

    def find_and_list_backups(self):
        try:
            all_backup_files = os.listdir(self.backup_dir)

            # Filtering files based on the pattern "TABLENAME[TIMESTAMP].json"
            filtered_files = [file for file in all_backup_files if re.match(r'.*\[\d{14}]\.json', file)]

            # Updating the listbox
            self.backup_files_listbox.delete(0, tk.END)
            for file in filtered_files:
                self.backup_files_listbox.insert(tk.END, file)
        except Exception as e:
            print("Error listing backup files:", e)

    def load_collection_data(self):
        collection_name = self.selected_table_entry.get()
        if collection_name:
            self.tree.delete(*self.tree.get_children())  # Clear existing data in the treeview
            mongo_client = MongoClient(zconstants.MONGO_URI)
            mongo_db = mongo_client[zconstants.MONGO_DATABASE_NAME]
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
        # Trigger backup for all collections
        if not self.loop.is_running():
            self.loop.run_until_complete(self.backup_all_collections())
        else:
            asyncio.run_coroutine_threadsafe(self.backup_all_collections(), self.loop)
            message = f"Backup of all tables."
            self.append_message(message)

    def on_backup_button_clicked(self):
        selected_collection = self.get_selected_collection()
        if selected_collection:
            # Ensure the event loop is running
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
            self.selected_backup_filename = event.widget.get(index)  # Store the selected filename

            # Correct way to update the text of an Entry widget
            self.selected_backup_file_entry.config(state='normal')  # Enable the widget for editing
            self.selected_backup_file_entry.delete(0, tk.END)  # Remove any existing text
            self.selected_backup_file_entry.insert(0, self.selected_backup_filename)  # Insert the new text
            self.selected_backup_file_entry.config(state='readonly')  # Set the widget back to readonly

            self.append_message(f"Selected backup file: {self.selected_backup_filename}")

    @staticmethod
    def on_entry_click(event):
        # Temporarily change the state to normal to enable text selection
        event.widget.config(state='normal')
        event.widget.select_range(0, tk.END)
        # Set the state back to readonly
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
        self.update_backup_files_list()  # Update the backup files list for the selected table

    def open_file_explorer(self):
        # Set initial directory to MONGO_BACKUP_DIR and file type to .json
        # backup_dir = os.path.join(PROJECT_DIR, MONGO_DIR)  # Construct the full backup directory path
        file_path = filedialog.askopenfilename(initialdir=self.backup_dir, filetypes=[("JSON files", "*.json")])
        if file_path:
            # Extract just the file name from the selected path
            file_name = os.path.basename(file_path)
            self.selected_backup_file_entry.config(state='normal')
            self.selected_backup_file_entry.delete(0, tk.END)
            self.selected_backup_file_entry.insert(0, file_name)
            self.selected_backup_file_entry.config(state='readonly')
            self.selected_backup_filename = file_name  # Update the selected backup filename attribute

    def read_backup_file(self, file_name):
        # backup_dir = os.path.join(PROJECT_DIR, MONGO_DIR)
        file_path = os.path.join(self.backup_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            self.append_message(f"Error reading file: {e}")
            return None

    async def restore_db(self):
        # Method for restoring the database from a backup
        selected_collection = self.get_selected_collection()
        selected_option = self.restore_options.get()

        if not self.selected_backup_filename:
            self.append_message("No file selected for restore")
            return

        data = self.read_backup_file(self.selected_backup_filename)
        if data is None:
            return

        # Preprocess the data
        data = self.preprocess_data(data)

        collection_name = selected_collection if selected_collection else self.selected_backup_filename.partition('[')[
            0]
        collection_to_restore = self.db[collection_name]

        try:
            if selected_option == "Add without Updating":
                inserted_count = await add_without_updating(collection_to_restore, data)
                self.append_message(
                    f"Restored {inserted_count} records to {collection_name} using 'Add without Updating'")
            # ... [Similar calls for other options]
        except Exception as e:
            self.append_message(f"Restore operation error: {str(e)}")

    @staticmethod
    def preprocess_data(data):
        for doc in data:
            # Rename 'user_id' to 'creator' if it exists
            if 'user_id' in doc:
                doc['creator'] = doc.pop('user_id')

            # Update 'created_by' to 'creator'
            if 'created_by' in doc:
                doc['creator'] = doc.pop('created_by')

            # Rename 'created_at' to 'date_time'
            if 'created_at' in doc:
                doc['date_time'] = doc.pop('created_at')

            # Convert 'creator' to ObjectId
            if 'creator' in doc and isinstance(doc['creator'], str):
                user = users_collection.find_one({'user_id': doc['creator']})
                doc['creator'] = user['_id'] if user else ObjectId()

        return data

    def run_async_tasks(self):
        # Schedule the async tasks
        run_async_in_tkinter(self.fetch_basic_info, self.loop)
        run_async_in_tkinter(self.update_table_list, self.loop)
        self.update_backup_files_list()
        self.after(10000, self.run_async_tasks)  # Schedule the next call

    # def run_flask_app(self):
    #     # Assuming flask_app.py is in the path "flask_backend/flask_app.py"
    #     # Directs output to the Flask output tab
    #     self.flask_process = self.run_program("flask_backend/flask_app.py", self.flask_output_text)
    #
    # def run_llm_runner(self):
    #     # Assuming llm_runner.py is in the path "runners/llm_runner.py"
    #     self.llm_process = self.run_program("runners/llm_runner.py", self.llm_output_text)
    #
    # def run_ocr_runner(self):
    #     # Assuming ocr_runner.py is in the path "runners/ocr_runner.py"
    #     self.ocr_process = self.run_program("runners/ocr_runner.py", self.ocr_output_text)

    def run_zmongo_retriever(self):
        # Assuming ocr_runner.py is in the path "runners/ocr_runner.py"
        self.zmongo_retriever_process = self.run_program("zmongo_retriever.py", self.zmongo_retriever_output_text)

    def run_program(self, program_path, output_widget=None):
        if output_widget not in self.current_output_widgets:
            self.current_output_widgets.append(output_widget)

            def target():
                process = subprocess.Popen(['python', program_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    if output_widget:
                        self.update_output_text(output_widget, line)
                    else:
                        print(line, end='')  # or handle it in another way
                process.wait()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            return thread  # Return the thread object instead of the process

    def setup_and_populate_treeview(self, records):
        # Assuming `self.tree` is your Treeview widget
        # First, clear any existing setup in the tree
        self.tree.delete(*self.tree.get_children())  # Clear the existing rows

        # Assuming `convert_mongo_to_metadata` converts a single record to a flat dict with unique paths as keys
        # and `get_value` fetches values based on these keys from the record
        if records:
            # Dynamically determine column names from the first record
            column_names = list(convert_json_to_metadata(records[0]).keys())

            # Reset the tree columns to these new column names
            self.tree["columns"] = column_names

            # Clear previous headings and columns setup
            for col in self.tree["columns"]:
                self.tree.heading(col, text=col, anchor='w')  # Use the actual column name for heading
                self.tree.column(col, anchor="w", width=120)  # Adjust width as needed

            # Populate the treeview with records
            for record in records:
                # Fetch values in the order of columns, considering the actual keys without modifications
                values = [get_value(record, col) for col in column_names]
                self.tree.insert("", "end", values=values)

    # def start_flask_app(self):
    #     threading.Thread(target=self.run_flask_app, daemon=True).start()
    #
    # def start_llm_runner(self):
    #     threading.Thread(target=self.run_llm_runner, daemon=True).start()
    #
    # def start_ocr_runner(self):
    #     threading.Thread(target=self.run_ocr_runner, daemon=True).start()

    def start_zmongo_retriever(self):
        threading.Thread(target=self.run_zmongo_retriever, daemon=True).start()

    def update_backup_files_list(self):
        selected_collection = self.get_selected_collection()
        if not selected_collection:
            return  # Exit if no collection is selected

        # backup_dir = os.path.join(PROJECT_DIR, MONGO_DIR)
        try:
            all_backup_files = os.listdir(self.backup_dir)

            # Filter files based on exact match before the timestamp
            filtered_files = []
            for file in all_backup_files:
                file_table_name, _, _ = file.partition('[')  # Split by the first '['
                if file_table_name == selected_collection:
                    filtered_files.append(file)

            # Sorting logic (remains the same as before)
            # ...

            # Updating the listbox
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
        # Fetch and update the table names
        try:
            collections = await self.db.list_collection_names()
            self.table_listbox.delete(0, tk.END)
            for collection in collections:
                self.table_listbox.insert(tk.END, collection)
        except Exception as e:
            print("Error updating table list:", e)


if __name__ == "__main__":
    app = SystemManagerGUI(mongo_uri=zconstants.MONGO_URI,
                           db_name=zconstants.MONGO_DATABASE_NAME,
                           project_dir=zconstants.PROJECT_PATH,
                           project_backup_dir=zconstants.MONGO_BACKUP_DIR,
                           mongo_backup_dir=zconstants.MONGO_DATABASE_NAME)

    # Start the asyncio event loop in a separate thread
    loop_thread = threading.Thread(target=app.loop.run_forever, daemon=True)
    loop_thread.start()

    app.mainloop()
