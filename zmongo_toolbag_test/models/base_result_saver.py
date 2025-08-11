# zmongo_toolbag/base_result_saver.py

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from bson.objectid import ObjectId


class BaseResultSaver(ABC):
    @abstractmethod
    async def save(
        self,
        collection_name: str,
        record_id: Union[str, ObjectId],
        field_name: str,
        generated_text: str,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Save the generated result to the storage backend."""
        pass  # pragma: no cover