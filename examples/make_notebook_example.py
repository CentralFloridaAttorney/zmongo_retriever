import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Prefer /mnt/data if writable, else use project-local fallback
preferred_dir = "/mnt/data"
fallback_dir = os.path.expanduser("~/PycharmProjects/zmongo_retriever/notebooks")
output_dir = preferred_dir if os.access(preferred_dir, os.W_OK) else fallback_dir
os.makedirs(output_dir, exist_ok=True)

notebook_path = os.path.join(output_dir, "ZMongo_Notebook_Demo.ipynb")

nb = new_notebook(cells=[
    new_markdown_cell("# 📘 ZMongo Usage Demo\n\nThis notebook demonstrates how to use the ZMongo async MongoDB utility."),
    new_code_cell("""\
import asyncio
from zmongo_toolbag.zmongo import ZMongo

zmongo = ZMongo()
"""),
    new_code_cell("""\
async def main():
    await zmongo.clear_cache()
    await zmongo.insert_document("users", {"name": "Alice"})
    doc = await zmongo.find_document("users", {"name": "Alice"})
    print(doc)

asyncio.run(main())"""),
])

with open(notebook_path, "w") as f:
    nbformat.write(nb, f)

print(f"✅ Notebook written to {notebook_path}")
