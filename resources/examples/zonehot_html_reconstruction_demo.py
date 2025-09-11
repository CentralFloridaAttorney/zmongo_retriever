# zonehot_html_demo.py
# Python 3.10+
#
# =============================================================================
# OneHotDB Formatted HTML Reconstruction GUI Demo
# =============================================================================
"""
This script provides a GUI to demonstrate the visually lossless reconstruction
capabilities of the OneHotDB library, now including rich text formatting.

Features:
- A GUI to select any HTML file.
- A status bar providing real-time feedback.
- A tabbed display:
  - "Browser View": Renders the reconstructed HTML, including bold, italics,
    underline, and complex capitalization.
  - "Source View": Displays the reconstructed HTML source code that was
    generated from the formatting masks.
- The verification step is now visual: if the browser view looks correct,
  the reconstruction was a success.

Dependencies:
- onehotdb.py (and its dependencies like beautifulsoup4)
- A running MongoDB instance configured via environment variables.
- tkhtmlview: For rendering the browser preview.
  Install with: pip install tkhtmlview beautifulsoup4
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
        self.root.title("OneHotDB Formatted HTML Reconstruction Demo")
        self.root.geometry("800x600")

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()

        # --- GUI Widgets ---
        self.select_button = Button(root, text="Select HTML File", command=self.on_select_file)
        self.select_button.pack(pady=10)

        self.status_label = Label(root, text="Please select an HTML file to begin.")
        self.status_label.pack(pady=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)

        self.browser_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.browser_frame, text='Browser View')
        self.html_view = HTMLLabel(self.browser_frame, html="<p>Select an HTML file to see it rendered here.</p>")
        self.html_view.pack(fill='both', expand=True)

        self.source_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.source_frame, text='Source View')
        self.text_area = ScrolledText(self.source_frame, wrap='word', font=("Courier New", 10))
        self.text_area.pack(fill='both', expand=True)

        demo_config = VocabConfig(remove_stopwords=False)
        self.db = ZOneHotDB(
            documents_collection_name="formatted_html_demo",
            repo=ZMongo(),
            config=demo_config
        )

    def run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_async_task(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def on_select_file(self):
        filepath = filedialog.askopenfilename(title="Select an HTML File", filetypes=(("HTML Files", "*.html"),("All Files", "*.*")))
        if filepath:
            self.schedule_async_task(self.process_file(Path(filepath)))

    def update_ui(self, status: str, content: str):
        def _update():
            self.status_label.config(text=status)
            self.text_area.delete('1.0', 'end')
            self.text_area.insert('1.0', content)
            self.html_view.set_html(content)
            self.notebook.select(self.browser_frame)

        self.root.after(0, _update)

    async def process_file(self, file_path: Path):
        """The core async logic for processing the selected file."""
        try:
            self.update_ui(f"Loading file: {file_path.name}...", "Loading...")
            original_text = file_path.read_text(encoding='utf-8', errors='ignore')
            doc_id = file_path.stem

            # --- 1. Encode and Store ---
            self.update_ui(f"Parsing and encoding '{doc_id}'...", "Encoding...")
            await self.db.encode_and_store_text(doc_id, original_text)
            self.update_ui("Encoding complete. Now reconstructing...", "Reconstructing...")

            # --- 2. Reconstruct ---
            indices = await self.db.get_encoded_indices(doc_id)
            masks = await self.db.get_capitalization_mask(doc_id)

            word_tasks = [self.db.lexicon.get_word_from_index(idx) for idx in indices]
            words = await asyncio.gather(*word_tasks)

            reconstructed_tokens = []
            for i, word in enumerate(words):
                if word is None: continue

                mask_parts = masks[i].split(',') if i < len(masks) else ['0']

                # Apply capitalization first
                reconstructed_token = word
                cap_mask = mask_parts[0]
                if cap_mask == 'T':
                    reconstructed_token = word.capitalize()
                elif cap_mask == 'U':
                    reconstructed_token = word.upper()
                elif cap_mask.startswith('I,'):
                    indices_str = cap_mask[2:]
                    upper_indices = {int(idx) for idx in indices_str.split(',')}
                    char_list = list(word)
                    for idx in upper_indices:
                        if idx < len(char_list): char_list[idx] = char_list[idx].upper()
                    reconstructed_token = "".join(char_list)

                # Apply formatting tags
                style_masks = set(mask_parts[1:])
                if 'B' in style_masks: reconstructed_token = f"<b>{reconstructed_token}</b>"
                if 'I' in style_masks: reconstructed_token = f"<i>{reconstructed_token}</i>"
                if 'U' in style_masks: reconstructed_token = f"<u>{reconstructed_token}</u>"

                reconstructed_tokens.append(reconstructed_token)

            reconstructed_html = "".join(reconstructed_tokens)

            # --- 3. Display ---
            status = f"✅ Success: Reconstructed '{file_path.name}' with formatting."
            self.update_ui(status, reconstructed_html)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            self.update_ui(f"Error: {e}", f"An error occurred:\n\n{e}")


def main():
    root = Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

