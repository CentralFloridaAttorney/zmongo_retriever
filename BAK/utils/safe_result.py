import logging
from typing import Any, Optional
from bson.objectid import ObjectId
import json

logging.basicConfig(level=logging.INFO)

class SafeResult:
    """
    Wraps all MongoDB results in a predictable, serializable object.
    Provides:
      - .success: True/False
      - .data: main result (dict/list/primitive)
      - .error: error string or None
      - .model_dump(): JSON-serializable dict output
      - .original(): original dict(s) with restored keys (for documents)
    """

    def __init__(
            self,
            data: Any = None,
            *,
            success: Optional[bool] = None,
            error: Optional[str] = None,
    ):
        self.success = success if success is not None else (error is None)
        self.error = error
        self.data = self._convert(data)

    @staticmethod
    def _convert(obj):
        # Convert ObjectIds and other BSON types to strings for safe serialization
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, dict):
            return {k: SafeResult._convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SafeResult._convert(x) for x in obj]
        return obj

    @classmethod
    def ok(cls, data: Any):
        return cls(data=data, success=True, error=None)

    @classmethod
    def fail(cls, error: str, data: Any = None):
        return cls(data=data, success=False, error=error)

    def model_dump(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }

    def __repr__(self):
        return f"SafeResult(success={self.success}, error={self.error!r}, data={str(self.data)[:300]})"

    def original(self):
        """
        Reconstitute original dict(s) with any key aliases restored (e.g. 'usecret' -> '_secret').
        Handles both dict and list of dict.
        """

        def restore_one(d):
            # Detect "__keymap" presence and restore original keys
            if isinstance(d, dict) and "__keymap" in d:
                keymap = d.pop("__keymap")
                for safe_k, orig_k in keymap.items():
                    if safe_k in d:
                        d[orig_k] = d.pop(safe_k)
            # Convert string _id back to ObjectId, if possible
            if isinstance(d, dict) and "_id" in d:
                try:
                    d["_id"] = ObjectId(d["_id"])
                except Exception:
                    pass
            return d

        data = self.data
        if isinstance(data, dict):
            return restore_one(data.copy())
        if isinstance(data, list):
            return [restore_one(x.copy()) if isinstance(x, dict) else x for x in data]
        return data

    def to_json(self):
        return json.dumps(self.model_dump(), default=str)

    @staticmethod
    def unwrap(result: 'SafeResult', *, quiet: bool = False):
        """
        Return result.original() on success or raise/return None on failure.

        Args:
            result (SafeResult): The SafeResult instance to unwrap
            quiet (bool): If True, log errors and return None instead of raising

        Returns:
            Any: The original data (with ObjectIds restored) from successful result or None if quiet=True and failed

        Raises:
            RuntimeError: If the result failed and quiet=False
        """
        if result.success:
            return result.original()
        if quiet:
            logging.info("Mongo error: %s", result.error)
            return None
        raise RuntimeError(result.error)
