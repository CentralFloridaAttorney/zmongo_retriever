import os
import glob
import json
import pandas as pd
from datetime import datetime
import re

def load_benchmark_files(prefix: str):
    json_files = glob.glob(f"{prefix}_*.json")
    data = []
    for file in json_files:
        try:
            with open(file, "r") as f:
                result = json.load(f)
                result["source_file"] = os.path.basename(file)

                match = re.search(r"(\d{8}_\d{6})", file)
                if match:
                    result["timestamp"] = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                else:
                    result["timestamp"] = None

                data.append(result)
        except Exception as e:
            print(f"Skipping file {file}: {e}")
    return pd.DataFrame(data)

def ensure_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    return df

def build_comparison_matrix(combined_df: pd.DataFrame) -> pd.DataFrame:
    latest_zmongo = combined_df[combined_df["engine"] == "ZMongo"].sort_values("timestamp", ascending=False).head(1)
    latest_zmagnum = combined_df[combined_df["engine"] == "ZMagnum"].sort_values("timestamp", ascending=False).head(1)

    if latest_zmongo.empty or latest_zmagnum.empty:
        print("❌ Not enough benchmark data for both engines to compare.")
        return pd.DataFrame()

    metrics = ["insert_time_sec", "update_time_sec", "field_fetch_time_sec", "index_recommend_time_sec", "inserted_count"]
