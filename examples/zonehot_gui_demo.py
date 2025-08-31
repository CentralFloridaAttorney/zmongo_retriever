# zonehot_html_demo.py
# Python 3.10+
#
# =============================================================================
# OneHotDB HTML Reconstruction GUI Demo
# =============================================================================
"""
This script provides a simple graphical user interface (GUI) to demonstrate the
lossless reconstruction capabilities of the OneHotDB library.

Features:
- A simple Tkinter GUI for user interaction.
- A "Select File" button to open a file dialog.
- A status bar to provide real-time feedback on the process.
- A tabbed display:
  - "Browser View": Renders the reconstructed HTML for a visual preview.
  - "Source View": Displays the raw, perfectly reconstructed source code.
- Automatically switches to the appropriate view based on file type.

This demo showcases the end-to-end process:
1. Loading a file from disk.
2. Encoding its content using OneHotDB and storing it in MongoDB.
3. Retrieving the encoded data and reconstructing the original text.
4. Verifying the reconstructed text matches the original.
5. Displaying the result to the user in the appropriate view.

Dependencies:
- onehotdb.py (and its dependencies like zmongo.py)
- A running MongoDB instance configured via environment variables.
- tkhtmlview: For rendering the browser preview.
  Install with: pip install tkhtmlview
"""

import asyncio
import logging
import threading
from pathlib import Path
from tkinter import Tk, Button, Label, filedialog, ttk
from tkinter.scrolledtext import ScrolledText

# --- Ensure correct imports for the library ---
try:
    from zmongo_toolbag.zonehotdb import ZOneHotDB, VocabConfig, ZMongo
except (ImportError, ModuleNotFoundError):
    from zmongo_toolbag.zonehotdb import ZOneHotDB, VocabConfig, ZMongo

# --- Import for the HTML view ---
try:
    from tkhtmlview import HTMLLabel
except ImportError:
    print("Error: tkhtmlview is not installed. Please run 'pip install tkhtmlview'")
    exit(1)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class App:
    """The main application class for the Tkinter GUI."""

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("OneHotDB Reconstruction Demo")
        self.root.geometry("800x600")

        # Set up the asyncio event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()

        # --- GUI Widgets ---
        self.select_button = Button(root, text="Select File (.html, .txt, etc.)", command=self.on_select_file)
        self.select_button.pack(pady=10)

        self.status_label = Label(root, text="Please select a file to begin.")
        self.status_label.pack(pady=5)

        # Create a Notebook to hold the different views (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)

        # Create a frame and widget for the browser view
        self.browser_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.browser_frame, text='Browser View')
        self.html_view = HTMLLabel(self.browser_frame, html="<p>Select an HTML file to see it rendered here.</p>")
        self.html_view.pack(fill='both', expand=True)

        # Create a frame and widget for the source view
        self.source_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.source_frame, text='Source View')
        self.text_area = ScrolledText(self.source_frame, wrap='word', font=("Courier New", 10))
        self.text_area.pack(fill='both', expand=True)

        # OneHotDB instance
        # Configure for perfect reconstruction (no stopwords, capitalization on).
        demo_config = VocabConfig(encode_capitalization=True, remove_stopwords=False)
        self.db = ZOneHotDB(
            documents_collection_name="html_demo_docs",
            repo=ZMongo(),
            config=demo_config
        )

    def run_async_loop(self):
        """Runs the asyncio event loop in the background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_async_task(self, coro):
        """Schedules a coroutine to be run on the background event loop."""
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def on_select_file(self):
        """Handles the 'Select File' button click."""
        filepath = filedialog.askopenfilename(
            title="Select a File",
            filetypes=(("HTML Files", "*.html"), ("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if filepath:
            self.schedule_async_task(self.process_file(Path(filepath)))

    def update_ui(self, status: str, content: str, is_html: bool):
        """Thread-safe method to update all GUI components at once."""

        def _update():
            self.status_label.config(text=status)

            # Always update the source view
            self.text_area.delete('1.0', 'end')
            self.text_area.insert('1.0', content)

            if is_html:
                self.html_view.set_html(content)
                self.notebook.select(self.browser_frame)  # Switch to browser tab
            else:
                self.html_view.set_html("<p><i>Not an HTML file. No browser preview available.</i></p>")
                self.notebook.select(self.source_frame)  # Switch to source tab

        self.root.after(0, _update)

    async def process_file(self, file_path: Path):
        """The core async logic for processing the selected file."""
        try:
            self.update_ui(f"Loading file: {file_path.name}...", "Loading...", False)
            original_text = file_path.read_text(encoding='utf-8', errors='ignore')
            doc_id = file_path.stem

            # --- 1. Encode and Store ---
            self.update_ui(f"Encoding and storing '{doc_id}' in the database...", "Encoding...", False)
            await self.db.encode_and_store_text(doc_id, original_text)
            self.update_ui("Encoding complete. Now reconstructing...", "Reconstructing...", False)

            # --- 2. Reconstruct ---
            indices = await self.db.get_encoded_indices(doc_id)
            mask = await self.db.get_capitalization_mask(doc_id)

            word_tasks = [self.db.lexicon.get_word_from_index(idx) for idx in indices]
            words = await asyncio.gather(*word_tasks)

            reconstructed_tokens = []
            for i, word in enumerate(words):
                if word is None: continue

                mask_value = mask[i] if i < len(mask) else '0'
                reconstructed_token = word

                if mask_value == 'T':
                    reconstructed_token = word.capitalize()
                elif mask_value == 'U':
                    reconstructed_token = word.upper()
                elif mask_value.startswith('I,'):
                    indices_str = mask_value[2:]
                    upper_indices = {int(idx) for idx in indices_str.split(',')}
                    char_list = list(word)
                    for idx in upper_indices:
                        if idx < len(char_list):
                            char_list[idx] = char_list[idx].upper()
                    reconstructed_token = "".join(char_list)

                reconstructed_tokens.append(reconstructed_token)

            reconstructed_text = "".join(reconstructed_tokens)

            # --- 3. Verify and Display ---
            is_html = file_path.suffix.lower() in ['.html', '.htm']
            if original_text == reconstructed_text:
                status = f"✅ SUCCESS: Perfectly reconstructed '{file_path.name}'."
            else:
                status = f"❌ FAILURE: Reconstruction mismatch for '{file_path.name}'."

            self.update_ui(status, reconstructed_text, is_html)

        except Exception as e:
            logger.error(f"An error occurred during file processing: {e}", exc_info=True)
            error_message = f"An error occurred:\n\n{e}"
            self.update_ui(f"Error: {e}", error_message, False)


def main():
    """Sets up and runs the Tkinter application."""
    root = Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

