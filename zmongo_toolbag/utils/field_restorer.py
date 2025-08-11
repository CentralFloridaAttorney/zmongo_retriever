from typing import Any, Callable, Dict, Optional, Type
from pydantic import AnyUrl, ValidationError
from bson import ObjectId
from datetime import datetime

class FieldRestorer:
    """
    Lightning-fast, class-based field restorer for Mongo→Pydantic pipelines.
    Allows registration of per-field converters (for AnyUrl, ObjectId, datetime, etc.).
    """
    _registry: Dict[str, Callable[[Any], Any]] = {}

    @classmethod
    def register(cls, field: str, converter: Callable[[Any], Any]) -> None:
        """Register a converter for a field name."""
        cls._registry[field] = converter

    @classmethod
    def restore(cls, d: Any) -> Any:
        """Recursively restores a dict, list, or single value using the field registry."""
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                # Recursively restore nested dicts/lists
                if isinstance(v, dict) or isinstance(v, list):
                    v = cls.restore(v)
                # Field-level conversion if registered
                if k in cls._registry:
                    v = cls._registry[k](v)
                out[k] = v
            return out
        elif isinstance(d, list):
            return [cls.restore(i) for i in d]
        else:
            return d

# --- Register default converters ---

def objectid_to_str(val: Any) -> Any:
    return str(val) if isinstance(val, ObjectId) else val

def url_to_anyurl(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return AnyUrl(val)
        except (ValidationError, TypeError, ValueError):
            return val
    return val

def parse_datetime_if_iso(val: Any) -> Any:
    if isinstance(val, str) and len(val) >= 19:
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return val
    return val

# Register core field converters
FieldRestorer.register("_id", objectid_to_str)
FieldRestorer.register("agent_url", url_to_anyurl)
FieldRestorer.register("fleet_server_url", url_to_anyurl)
FieldRestorer.register("last_seen_at", parse_datetime_if_iso)
# Add more as needed: FieldRestorer.register("your_field", your_converter)

# --- Usage with SafeResult ---

# In your as_pydantic call:
# agent_obj = result.as_pydantic(FleetAgent, restore_dict_fn=FieldRestorer.restore)

