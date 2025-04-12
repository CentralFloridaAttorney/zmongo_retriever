# zmongo_toolbag/openai_result_saver.py

from typing import Optional, Union, Any
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.models.base_result_saver import BaseResultSaver


class OpenAIResultSaver(BaseResultSaver):
    def __init__(self, zmongo: Optional[ZMongo] = None):
        self.zmongo = zmongo or ZMongo()
        self._should_close = zmongo is None

    async def save(
        self,
        collection_name: str,
        record_id: Union[str, ObjectId],
        field_name: str,
        generated_text: str,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not generated_text or not field_name:
            raise ValueError("Generated text and field name must be provided.")

        if isinstance(record_id, str):
            record_id = ObjectId(record_id)

        update_data = {"$set": {field_name: generated_text}}
        if extra_fields:
            update_data["$set"].update(extra_fields)

        try:
            result = await self.zmongo.update_document(
                collection=collection_name,
                query={"_id": record_id},
                update_data=update_data,
                upsert=True,
            )
            return result.matched_count > 0 or result.upserted_id is not None
        finally:
            if self._should_close:
                self.zmongo.mongo_client.close()
