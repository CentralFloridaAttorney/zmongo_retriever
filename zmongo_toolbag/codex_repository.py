import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from bson import ObjectId
from bson.errors import InvalidId

from zmongo_toolbag.zmongo import ZMongo, SafeResult

logger = logging.getLogger(__name__)


class CodexRepository:
    """
    High-level repository wrapper around ZMongo for managing application documents.
    Provides robust, SafeResult-based CRUD with automatic ID conversion, backups,
    and health utilities.
    """

    def __init__(self, backup_root: Optional[Path] = None, codex_collection: str = "legal_codex"):
        self.db = ZMongo()
        self.backup_root = backup_root or (Path.home() / ".resources" / "backups")
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.codex_collection = codex_collection
        logger.info(f"✅ CodexRepository initialized using collection '{self.codex_collection}'")

    def _handle_objectid(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not data or "_id" not in data or not isinstance(data["_id"], str):
            return data
        try:
            data_copy = data.copy()
            data_copy["_id"] = ObjectId(data_copy["_id"])
            return data_copy
        except InvalidId:
            return data

    def _safe_call(self, func_name: str, *args, **kwargs) -> SafeResult:
        """Uniform SafeResult wrapper for all ZMongo calls."""
        try:
            func = getattr(self.db, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"[CodexRepository] Error in {func_name}: {e}", exc_info=True)
            return SafeResult.fail(f"Repository error in {func_name}: {e}", exc=e)

    # --- Basic CRUD ---
    def insert(self, collection: str, doc: Dict[str, Any]) -> SafeResult:
        return self._safe_call("insert_one", collection, doc)

    def find_one(self, collection: str, query: Dict[str, Any]) -> SafeResult:
        query = self._handle_objectid(query)
        return self._safe_call("find_one", collection, query)

    # --- FIX: Added the 'projection' parameter to the method signature ---
    def find_all(self, collection: str, query: Optional[Dict[str, Any]] = None,
                 projection: Optional[Dict[str, Any]] = None, limit: int = 1000) -> SafeResult:
        """Finds multiple documents, with optional projection to limit fields."""
        query = self._handle_objectid(query or {})
        # --- FIX: Pass the 'projection' argument to the underlying database call ---
        return self._safe_call("find_many", collection, query, projection=projection, limit=limit)

    def update(self, collection: str, query: Dict[str, Any], update_doc: Dict[str, Any]) -> SafeResult:
        query = self._handle_objectid(query)
        if not any(k.startswith("$") for k in update_doc.keys()):
            update_doc = {"$set": update_doc}
        return self._safe_call("update_one", collection, query, update_doc)

    def delete_one(self, collection: str, query: Dict[str, Any]) -> SafeResult:
        query = self._handle_objectid(query)
        return self._safe_call("delete_one", collection, query)

    def list_collections(self) -> SafeResult:
        return self._safe_call("list_collections")

    # --- Codex-Specific Logic ---
    def save_codex(self, codex: Dict[str, Any]) -> SafeResult:
        codex_id = codex.get("_id")
        if not codex_id:
            return SafeResult.fail("Cannot save codex without an '_id' field.")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_root / f"{codex_id}_{timestamp}.json"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(codex, f, indent=2, default=str)
            logger.info(f"💾 Codex backup saved to {backup_path}")

            query = self._handle_objectid({"_id": codex_id})
            return self._safe_call("insert_or_update", self.codex_collection, query, codex)
        except Exception as e:
            return SafeResult.fail(f"save_codex failed: {e}", exc=e)

    def get_all_codex_summaries(self) -> SafeResult:
        # This call will now work correctly because find_all accepts 'projection'.
        result = self.find_all(self.codex_collection, projection={"meta_title": 1, "modified_at": 1, "_id": 1})
        if not result.success:
            return result

        summaries = result.data or []
        summaries.sort(key=lambda x: x.get("modified_at") or datetime.min, reverse=True)
        return SafeResult.ok(summaries)

    def load_codex(self, codex_id: str) -> SafeResult:
        return self.find_one(self.codex_collection, {"_id": codex_id})