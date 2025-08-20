"""
Data Processing Module
=======================

This module provides a suite of utility functions and classes for cleaning,
converting, and exploring data structures.

- SafeResult: A robust wrapper for operation outcomes, now with advanced
  data discovery methods like .get(), .to_json(), and .to_metadata().
- DataProcessor: A collection of static methods for handling complex data
  types, flattening nested structures, and extracting values.
"""
import html
import json
import logging
from typing import Any, Dict, List, Union, Optional

import numpy as np
import pandas as pd
from bson.objectid import ObjectId
from bs4 import BeautifulSoup
from datetime import datetime
from collections import deque

# Configure module-level logger
logger = logging.getLogger(__name__)


class SafeResult:
    """
    A predictable, serializable wrapper for operation results, now enhanced
    with powerful data discovery and formatting methods.
    """

    def __init__(self, data: Any = None, *, success: bool, error: Optional[str] = None,
                 original_exc: Optional[Exception] = None, metadata_keymap: Optional[Dict[str, str]] = None):
        self.success = success
        self.error = error
        self.data = self._convert_bson(data)
        self._original_exc = original_exc
        self.metadata_keymap = metadata_keymap or {}

    @staticmethod
    def _convert_bson(obj: Any) -> Any:
        if isinstance(obj, ObjectId): return str(obj)
        if isinstance(obj, dict): return {k: SafeResult._convert_bson(v) for k, v in obj.items()}
        if isinstance(obj, list): return [SafeResult._convert_bson(x) for x in obj]
        return obj

    @classmethod
    def ok(cls, data: Any = None, **kwargs) -> 'SafeResult':
        return cls(data=data, success=True, **kwargs)

    @classmethod
    def fail(cls, error: str, data: Any = None, exc: Optional[Exception] = None, **kwargs) -> 'SafeResult':
        return cls(data=data, success=False, error=error, original_exc=exc, **kwargs)



    def model_dump(self) -> Dict[str, Any]:
        """
        Lightweight, pydantic-style export used by tests.
        """
        return {"success": self.success, "error": self.error, "data": self.data}

    def original(self) -> Any:
        """
        Reconstruct original data:
        - Convert stringified ObjectIds back to ObjectId
        - Apply top-level __keymap (e.g., {"usecret": "_secret"})
        - Handle lists of docs
        - For primitives, just return the data
        """
        if not self.success:
            # Preserve previous behavior for failures
            return self._original_exc

        data = self.data

        def _restore(doc: Any) -> Any:
            # primitives: return as-is
            if not isinstance(doc, (dict, list)):
                return doc

            if isinstance(doc, list):
                return [_restore(item) for item in doc]

            # dict case
            d = dict(doc)  # shallow copy
            # pull out keymap if present
            keymap = d.pop("__keymap", {})

            # restore _id if it looks like an ObjectId
            if "_id" in d and isinstance(d["_id"], str) and ObjectId.is_valid(d["_id"]):
                d["_id"] = ObjectId(d["_id"])

            # apply keymap translations (safe_key -> original_key)
            for safe_key, original_key in keymap.items():
                if safe_key in d:
                    d[original_key] = d.pop(safe_key)

            return d

        return _restore(data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a nested value from the result data using a dot-separated key.

        Example:
            >>> result = SafeResult.ok({"casebody": {"data": {"opinions": [{"text": "This is an opinion."}]}}})
            >>> result_text = result.get("casebody.data.opinions.0.text")
            >>> print(result_text)
            This is an opinion.
        """
        if not self.success or not isinstance(self.data, (dict, list)):
            return default
        return DataProcessor.get_value(self.data, key) or default

    def to_json(self, indent: int = 4) -> str:
        """
        Serializes the .data attribute to a formatted JSON string.
        """
        if not self.success or self.data is None:
            return json.dumps({"error": self.error, "success": False}, indent=indent)
        return json.dumps(self.data, indent=indent)

    def to_metadata(self) -> Dict[str, Any]:
        """
        Flattens the result data into a single-level dictionary and applies
        the metadata keymap to rename keys for clarity.
        """
        if not self.success or self.data is None:
            return {}

        flat_data = DataProcessor.flatten_json(self.data)

        if not self.metadata_keymap:
            return flat_data

        metadata = {}
        for raw_key, value in flat_data.items():
            friendly_key = self.metadata_keymap.get(raw_key, raw_key)
            metadata[friendly_key] = value

        return metadata

    def __repr__(self):
        return f"SafeResult(success={self.success}, error='{self.error}', data_preview='{str(self.data)[:100]}...')"


class DataProcessor:
    @staticmethod
    def get_value(json_data: Union[Dict[str, Any], List[Any]], key: str) -> Any:
        """
        Retrieves a value from a nested dictionary or list using a dot-separated key.
        """
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
    def flatten_json(json_obj: Any, prefix: str = "") -> Dict[str, Any]:
        """
        Flattens a nested dictionary or list into a single-level dictionary.
        """
        flat_dict: Dict[str, Any] = {}
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                flat_dict.update(DataProcessor.flatten_json(value, full_key))
        elif isinstance(json_obj, list):
            for idx, item in enumerate(json_obj):
                full_key = f"{prefix}.{idx}" if prefix else str(idx)
                flat_dict.update(DataProcessor.flatten_json(item, full_key))
        else:
            flat_dict[prefix] = json_obj
        return flat_dict

    # ... [ The rest of your DataProcessor static methods remain here ] ...
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
                except Exception:
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
        if isinstance(input_data, str):
            data_value = input_data
        elif isinstance(input_data, dict):
            converted = DataProcessor.convert_object_to_json(input_data)
            data_value = converted.get("output_text")
            if not isinstance(data_value, str):
                raise ValueError("Dictionary input must contain an 'output_text' key with a string value.")
        else:
            raise ValueError("Input to convert_text_to_html must be either a string or a dictionary.")

        soup = BeautifulSoup(data_value, "html.parser")
        for text_node in soup.find_all(string=True):
            text_node.replace_with(html.unescape(text_node))
        pretty_html = soup.prettify()
        return pretty_html
