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
import json
import logging
from typing import Any, Dict, Optional

from bson.objectid import ObjectId

from zmongo_toolbag.data_processing import DataProcessor

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
        _sentinel = object()
        val = DataProcessor.get_value(self.data, key)
        return default if val is None else val

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