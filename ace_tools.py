# ace_tools.py
import pandas as pd
import json
from typing import Optional

def display_dataframe_to_user(name: str, dataframe: pd.DataFrame, max_rows: int = 25) -> str:
    """Formats a DataFrame for user-facing display or JSON embedding."""
    preview = dataframe.head(max_rows)
    return f"Data Preview: {name}\n{preview.to_string(index=False)}"

def dataframe_to_json_snippet(dataframe: pd.DataFrame, max_rows: int = 10) -> str:
    """Returns a compact JSON string of the top N rows of the DataFrame."""
    limited_df = dataframe.head(max_rows)
    return json.dumps(limited_df.to_dict(orient="records"), indent=2, default=str)
