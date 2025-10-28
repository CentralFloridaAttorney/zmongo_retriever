import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from zmongo_toolbag.zmongo import ZMongo, SafeResult

logger = logging.getLogger(__name__)


class CodexRepository:
    """
    High-level repository wrapper around ZMongo for managing Codex documents.
    Provides SafeResult-based CRUD, backup, and health utilities.
    """

    def __init__(self, backup_root: Optional[Path] = None, codex_collection: str = "legal_codex"):
        self.db = ZMongo()
        self.backup_root = backup_root or (Path.home() / ".resources" / "backups")
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.codex_collection = codex_collection
        logger.info(f"✅ CodexRepository initialized using collection '{self.codex_collection}'")

    # ----------------------------------------------------------------------
    # Internal wrapper for safety
    # ----------------------------------------------------------------------
    def _safe_call(self, func_name: str, *args, **kwargs) -> SafeResult:
        """Uniform SafeResult wrapper for all ZMongo calls."""
        try:
            func = getattr(self.db, func_name)
            result = func(*args, **kwargs)
            if not isinstance(result, SafeResult):
                return SafeResult.ok(result)
            return result
        except Exception as e:
            logger.error(f"[CodexRepository] Error calling {func_name}: {e}", exc_info=True)
            return SafeResult.fail(f"CodexRepository error in {func_name}: {e}")

    # ----------------------------------------------------------------------
    # Basic CRUD
    # ----------------------------------------------------------------------
    def insert(self, collection: str, doc: Dict[str, Any]) -> SafeResult:
        return self._safe_call("insert_one", collection, doc)

    def find_one(self, collection: str, query: Dict[str, Any]) -> SafeResult:
        return self._safe_call("find_one", collection, query)

    def find_all(self, collection: str, query: Optional[Dict[str, Any]] = None, limit: int = 1000) -> SafeResult:
        return self._safe_call("find_many", collection, query or {}, limit=limit)

    def update(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> SafeResult:
        """Update using $set semantics."""
        if not any(k.startswith("$") for k in update.keys()):
            update = {"$set": update}
        return self._safe_call("update_one", collection, query, update)

    def delete(self, collection: str, query: Dict[str, Any]) -> SafeResult:
        return self._safe_call("delete_one", collection, query)

    def delete_all_documents(self, collection: str) -> SafeResult:
        return self._safe_call("delete_all_documents", collection)

    def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> SafeResult:
        return self._safe_call("aggregate", collection, pipeline)

    def list_collections(self) -> SafeResult:
        return self._safe_call("list_collections")

    def insert_or_update(self, collection: str, query_or_doc: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> SafeResult:
        """Unified insert or upsert call using ZMongo’s SafeResult API."""
        return self._safe_call("insert_or_update", collection, query_or_doc, data)

    # ----------------------------------------------------------------------
    # Codex-specific logic
    # ----------------------------------------------------------------------
    def save_codex(self, codex: Dict[str, Any]) -> SafeResult:
        """
        Save or update a Codex document.
        Automatically backs up the Codex as JSON under ~/.resources/backups/
        """
        try:
            codex_id = codex.get("_id", "unknown_codex")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_root / f"{codex_id}_{timestamp}.json"

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(codex, f, indent=2, default=str)
            logger.info(f"💾 Codex backup saved to {backup_path}")

            result = self.db.insert_or_update(self.codex_collection, {"_id": codex_id}, codex)
            return result
        except Exception as e:
            logger.error(f"Failed to save codex: {e}", exc_info=True)
            return SafeResult.fail(f"save_codex failed: {e}")

    def get_all_codex_summaries(self) -> SafeResult:
        """Return summarized metadata for all codex documents."""
        result = self.find_all(self.codex_collection)
        if not result.success:
            return result

        docs = result.data or []
        summaries = [
            {
                "_id": d.get("_id"),
                "meta_title": d.get("meta_title", "Untitled Codex"),
                "created_at": d.get("created_at"),
                "modified_at": d.get("modified_at"),
            }
            for d in docs
        ]

        # FIX: Provide a default datetime object to prevent sorting mixed types.
        summaries.sort(
            key=lambda x: x.get("modified_at") or x.get("created_at") or datetime.min,
            reverse=True,
        )
        return SafeResult.ok(summaries)

    def load_codex(self, codex_id: str) -> SafeResult:
        """Retrieve a Codex document by ID."""
        return self.find_one(self.codex_collection, {"_id": codex_id})

    def set_codex_collection(self, name: str):
        """Switch the active Codex collection."""
        if not name:
            logger.warning("Empty collection name ignored.")
            return
        self.codex_collection = name
        logger.info(f"Codex collection changed to '{name}'")

    # ----------------------------------------------------------------------
    # Backup and Export Utilities
    # ----------------------------------------------------------------------
    def export_collection(self, collection_name: str, file_path: Path) -> SafeResult:
        """Export all documents from a collection to a JSON file."""
        try:
            result = self.find_all(collection_name, limit=0)
            if not result.success:
                return result

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result.data, f, indent=2, default=str)
            return SafeResult.ok({
                "exported_count": len(result.data),
                "path": str(file_path),
            })
        except Exception as e:
            return SafeResult.fail(f"Export failed: {e}")

    def backup_all_collections(self) -> SafeResult:
        """Backup all MongoDB collections to a timestamped folder."""
        collections_result = self.list_collections()
        if not collections_result.success:
            return collections_result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_root / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for coll in collections_result.data:
            file_path = backup_dir / f"{coll}.json"
            export_result = self.export_collection(coll, file_path)
            if not export_result.success:
                return export_result

        return SafeResult.ok({
            "backup_path": str(backup_dir),
            "collection_count": len(collections_result.data),
        })

    # ----------------------------------------------------------------------
    # Health and Maintenance
    # ----------------------------------------------------------------------
    def sync_timestamp(self) -> SafeResult:
        """
        Health check: get MongoDB server time or fallback to local UTC.
        Ensures connection and loop are healthy.
        """
        try:
            res = self._safe_call("run_command", {"isMaster": 1})
            if not res.success:
                return res

            server_time = res.data.get("localTime") if isinstance(res.data, dict) else None
            if server_time is None:
                server_time = datetime.now()
            return SafeResult.ok({"server_time": server_time})
        except Exception as e:
            logger.error(f"sync_timestamp failed: {e}", exc_info=True)
            return SafeResult.fail(f"sync_timestamp failed: {e}")

    def health_summary(self) -> SafeResult:
        """
        Combined health check returning connection status, collections,
        and count of documents in the Codex collection.
        """
        try:
            ts = self.sync_timestamp()
            if not ts.success:
                return ts

            collections = self.list_collections()
            if not collections.success:
                return collections

            count = self.db.count_documents(self.codex_collection, {})
            if not count.success:
                return count

            return SafeResult.ok({
                "server_time": ts.data["server_time"],
                "collections": collections.data,
                "codex_collection": self.codex_collection,
                "codex_count": count.data,
                "status": "healthy",
            })
        except Exception as e:
            return SafeResult.fail(f"health_summary failed: {e}")

    def close(self) -> SafeResult:
        """Close MongoDB client cleanly."""
        return self._safe_call("close")
