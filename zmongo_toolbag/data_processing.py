"""
Data Processing Module
=======================

This module provides a suite of utility functions for cleaning and converting data,
flattening nested structures, extracting nested values with dot-separated keys,
and formatting text as HTML.

It is designed to handle MongoDB documents (including ObjectId and datetime),
pandas DataFrames, numpy arrays, and other complex objects. All functions are
implemented as static methods under the DataProcessing class.

If you need to control any behavior via environment variables, please add the
appropriate keys to your .env file and update the code accordingly.
"""
import ast
import html
import json
import logging
import re
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from bson import ObjectId
from bs4 import BeautifulSoup
from bson.errors import InvalidId

# Configure module-level logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

class DataProcessing:
    @staticmethod
    def clean_output_text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        cleaned_text = text.strip()
        if cleaned_text.startswith("```html"):
            cleaned_text = cleaned_text[len("```html"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        return cleaned_text

    @staticmethod
    def convert_object_to_json(data: Any) -> Any:
        def convert(obj: Any, seen: set, depth: int = 0) -> Any:
            if depth > 100:
                return {"__error__": "Maximum depth exceeded"}
            obj_id = id(obj)
            if obj_id in seen:
                return {"__circular_reference__": obj.__class__.__name__}
            if isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            seen.add(obj_id)
            if isinstance(obj, (list, tuple, deque, set)):
                return [convert(item, seen, depth + 1) for item in obj]
            if isinstance(obj, dict):
                return {str(key): convert(value, seen, depth + 1) for key, value in obj.items()}
            if isinstance(obj, ObjectId):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                try:
                    return convert(obj.to_dict(), seen, depth + 1)
                except Exception as e:
                    # logger.warning(f"Error converting via to_dict: {e}")
                    return str(obj)
            if hasattr(obj, "__dict__"):
                obj_attrs = {
                    attr: convert(getattr(obj, attr), seen, depth + 1)
                    for attr in dir(obj)
                    if not attr.startswith("_") and not callable(getattr(obj, attr))
                }
                if obj_attrs:
                    return obj_attrs
            return str(obj)

        return convert(data, seen=set())

    @staticmethod
    def convert_text_to_html(input_data: Union[str, Dict[str, Any]]) -> str:
        # logger.info(f"convert_text_to_html input_data: {input_data}")

        if isinstance(input_data, str):
            data_value = input_data
        elif isinstance(input_data, dict):
            converted = DataProcessing.convert_object_to_json(input_data)
            data_value = converted.get("output_text")
            if not isinstance(data_value, str):
                raise ValueError("Dictionary input must contain an 'output_text' key with a string value.")
        else:
            raise ValueError("Input to convert_text_to_html must be either a string or a dictionary.")

        # logger.info(f"convert_text_to_html data_value: {data_value}")
        soup = BeautifulSoup(data_value, "html.parser")
        for text_node in soup.find_all(string=True):
            text_node.replace_with(html.unescape(text_node))
        pretty_html = soup.prettify()
        return pretty_html

    @staticmethod
    def detect_unicode_surrogates(text: str) -> bool:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        surrogate_pattern = re.compile(r"[\uD800-\uDBFF][\uDC00-\uDFFF]")
        return bool(surrogate_pattern.search(text))

    @staticmethod
    def convert_json_to_metadata(json_object: Union[Dict[str, Any], List[Any]],
                                 existing_metadata: Dict[str, Any] = None,
                                 metadata_prefix: str = "") -> Dict[str, Any]:
        if existing_metadata is None:
            existing_metadata = {}

        if isinstance(json_object, dict):
            for key, value in json_object.items():
                new_prefix = f"{metadata_prefix}.{key}" if metadata_prefix else key
                DataProcessing.convert_json_to_metadata(value, existing_metadata, new_prefix)
        elif isinstance(json_object, list):
            for idx, item in enumerate(json_object):
                item_prefix = f"{metadata_prefix}.{idx}" if metadata_prefix else str(idx)
                DataProcessing.convert_json_to_metadata(item, existing_metadata, item_prefix)
        else:
            existing_metadata[metadata_prefix] = json_object
        return existing_metadata

    @staticmethod
    def convert_mongo_to_metadata(dict_data: Dict[str, Any],
                                  existing_metadata: Dict[str, Any] = None,
                                  metadata_prefix: str = "") -> Dict[str, Any]:
        if existing_metadata is None:
            existing_metadata = {}
        if dict_data is None:
            return existing_metadata

        if isinstance(dict_data, dict):
            for key, value in dict_data.items():
                new_prefix = f"{metadata_prefix}_{key}" if metadata_prefix else key
                DataProcessing.convert_mongo_to_metadata(value, existing_metadata, new_prefix)
        elif isinstance(dict_data, list):
            for idx, item in enumerate(dict_data):
                item_prefix = f"{metadata_prefix}_{idx}" if metadata_prefix else str(idx)
                if isinstance(item, (dict, list)):
                    DataProcessing.convert_mongo_to_metadata(item, existing_metadata, item_prefix)
                else:
                    existing_metadata[item_prefix] = item
        else:
            existing_metadata[metadata_prefix] = dict_data

        return existing_metadata


    @staticmethod
    def flatten_json(json_obj: Any, prefix: str = "") -> Dict[str, Any]:
        flat_dict: Dict[str, Any] = {}
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                flat_dict.update(DataProcessing.flatten_json(value, full_key))
        elif isinstance(json_obj, list):
            for idx, item in enumerate(json_obj):
                full_key = f"{prefix}.{idx}" if prefix else str(idx)
                flat_dict.update(DataProcessing.flatten_json(item, full_key))
        else:
            flat_dict[prefix] = json_obj
        return flat_dict

    @staticmethod
    def get_value(json_data: Dict[str, Any], key: str) -> Any:
        keys = key.split(".")
        value = json_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            elif isinstance(value, list) and k.isdigit():
                index = int(k)
                value = value[index] if 0 <= index < len(value) else None
            else:
                return None
            if value is None:
                return None
        return value

    @staticmethod
    def get_keys_from_json(json_object: Dict[str, Any]) -> List[str]:
        flat = DataProcessing.flatten_json(json_object)
        return list(flat.keys())

    @staticmethod
    def get_oid_value(value):
        """
        Attempts to extract and return a valid ObjectId from various formats.
        """
        if value is None:
            return None

        if isinstance(value, ObjectId):
            return value

        if isinstance(value, str):
            value = value.strip()
            if value.startswith("{") and value.endswith("}"):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass

        if isinstance(value, dict) and '$oid' in value:
            value = value['$oid']

        try:
            return ObjectId(value)
        except (InvalidId, TypeError):
            return None

    @staticmethod
    def get_opinion_from_zcase(zcase: Dict[str, Any]) -> str:
        try:
            casebody = zcase.get("casebody", {})
            data = casebody.get("data", {})
            opinions = data.get("opinions", [])
            if opinions and isinstance(opinions, list):
                first_opinion = opinions[0]
                return first_opinion.get("text", "No opinion text found.")
            else:
                return "No opinions found."
        except Exception as e:
            # logger.error(f"Error extracting opinion from zcase: {e}")
            return f"Error extracting opinion: {e}"

    @staticmethod
    def get_values_as_list(df: pd.DataFrame, prefix: str = "embedding_") -> List[Any]:
        embedding_columns = [col for col in df.columns if col.startswith(prefix)]
        embeddings = df[embedding_columns].values.tolist()
        return [item for sublist in embeddings for item in sublist]

    @staticmethod
    def convert_object_to_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: DataProcessing.convert_object_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataProcessing.convert_object_to_serializable(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {attr: DataProcessing.convert_object_to_serializable(getattr(obj, attr))
                    for attr in dir(obj)
                    if not attr.startswith("_") and not callable(getattr(obj, attr))}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    @staticmethod
    def list_to_string(items: List[Any]) -> str:
        return "".join(map(str, items))
