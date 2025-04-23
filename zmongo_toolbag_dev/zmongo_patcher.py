import os
import re
from pathlib import Path

import ace_tools

SCRIPTS_DIR = Path("scripts")
PATCHED_COMMENT = "# ZMongo patch applied"

def patch_script_file(file_path: Path):
    with open(file_path, "r") as f:
        code = f.read()

    if PATCHED_COMMENT in code:
        return False  # already patched

    # 1. Patch the __init__ method to accept zmongo=None and set self.db accordingly
    init_pattern = re.compile(r"def __init__\(self(.*?)\):", re.DOTALL)
    match = init_pattern.search(code)

    if not match:
        return False  # no init found, skip

    original_args = match.group(1).strip()
    new_args = original_args + ", zmongo: Optional[ZMongo] = None" if original_args else "zmongo: Optional[ZMongo] = None"
    patched_init = f"def __init__(self, {new_args}):\n        {PATCHED_COMMENT}\n        self.db = zmongo or ZMongo()"

    # Replace the original __init__ line
    start = match.start()
    end = match.end()
    code = code[:start] + patched_init + code[end:]

    # 2. Ensure required import is added
    if "from zmongo_toolbag.zmongo import ZMongo" not in code:
        code = f"from zmongo_toolbag.zmongo import ZMongo\nfrom typing import Optional\n\n{code}"

    with open(file_path, "w") as f:
        f.write(code)

    return True

# Apply the patch to all .py files in the scripts directory
patched_files = []
for py_file in SCRIPTS_DIR.glob("*.py"):
    if patch_script_file(py_file):
        patched_files.append(py_file.name)

import pandas as pd

ace_tools.display_dataframe_to_user("Patched Python Scripts", pd.DataFrame(patched_files, columns=["Patched Script Files"]))
