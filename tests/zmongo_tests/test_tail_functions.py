# tests/zmongo_tests/test_tail_functions.py
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoTailFunctions(unittest.IsolatedAsyncioTestCase):
    """
    Validate the five utility helpers at the “tail” of ZMongo:
        • list_collections
        • count_documents
        • get_document_by_id
        • get_field_names
        • log_training_metrics

    NOTE ─ SafeResult is assumed to expose:
        ├── sr["ok"]       → bool      success-flag
        ├── sr["payload"]  → Any       main data (may be 0/[]/None on failure)
        └── sr["error"]    → str | None
    """

    # ────────────────────────────────────────────── life-cycle
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.coll = "test_tail_funcs"
        await self.repo.delete_all_documents(self.coll)

    async def asyncTearDown(self):
        await self.repo.delete_all_documents(self.coll)
        await self.repo.close()

    # ────────────────────────────────────────────── list_collections
    async def test_list_collections_success_and_error(self):
        ok = await self.repo.list_collections()
        self.assertTrue(ok["ok"])
        self.assertIsInstance(ok["payload"], list)

        # force an exception
        with patch.object(
            self.repo.db, "list_collection_names",
            AsyncMock(side_effect=RuntimeError("kaput")),
        ):
            err = await self.repo.list_collections()
            self.assertFalse(err["ok"])
            self.assertEqual(err["payload"], [])
            self.assertIsNotNone(err["error"])

    # ────────────────────────────────────────────── count_documents
    async def test_count_documents(self):
        await self.repo.insert_documents(
            self.coll, [{"_id": ObjectId(), "val": i} for i in range(3)]
        )

        good = await self.repo.count_documents(self.coll)
        self.assertTrue(good["ok"])
        self.assertGreaterEqual(good["payload"], 3)

        # patch the class-level coroutine so *any* collection instance fails
        cls = self.repo.db[self.coll].__class__
        with patch.object(
            cls, "estimated_document_count",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            bad = await self.repo.count_documents(self.coll)
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["payload"], 0)
            self.assertIsNotNone(bad["error"])

    # ────────────────────────────────────────────── get_document_by_id
    async def test_get_document_by_id(self):
        doc_id = ObjectId()
        await self.repo.insert_document(self.coll, {"_id": doc_id, "foo": "bar"})

        # happy path
        found = await self.repo.get_document_by_id(self.coll, str(doc_id))
        self.assertEqual(found["payload"]["_id"], str(doc_id))

        # invalid string
        bad_id = await self.repo.get_document_by_id(self.coll, "not-an-oid")
        self.assertTrue(bad_id["ok"])
        self.assertIsNone(bad_id["payload"])

        # id not found
        missing = await self.repo.get_document_by_id(self.coll, str(ObjectId()))
        self.assertTrue(missing["ok"])
        self.assertIsNone(missing["payload"])

        # error branch (patch find_one at the class level)
        cls = self.repo.db[self.coll].__class__
        with patch.object(
            cls, "find_one",
            AsyncMock(side_effect=RuntimeError("db-fail")),
        ):
            err = await self.repo.get_document_by_id(self.coll, str(doc_id))
            self.assertFalse(err["ok"])
            self.assertIsNone(err["payload"])
            self.assertIsNotNone(err["error"])

    # ────────────────────────────────────────────── get_field_names
    async def test_get_field_names(self):
        empty = await self.repo.get_field_names(self.coll)
        self.assertEqual(empty["payload"], [])

        await self.repo.insert_document(self.coll, {"field1": 1, "field2": 2})
        fields = await self.repo.get_field_names(self.coll)
        self.assertCountEqual(fields["payload"], ["field1", "field2"])

        cls = self.repo.db[self.coll].__class__
        with patch.object(
            cls, "find",
            AsyncMock(side_effect=RuntimeError("fail")),
        ):
            err = await self.repo.get_field_names(self.coll)
            self.assertFalse(err["ok"])
            self.assertEqual(err["payload"], [])

    # ────────────────────────────────────────────── log_training_metrics
    def test_log_training_metrics(self):
        good = self.repo.log_training_metrics({"acc": 0.99})
        self.assertTrue(good["ok"])
        self.assertTrue(good["payload"])
        with patch.object(
                self.repo.sync_db["training_metrics"],
                "insert_one",
                MagicMock(side_effect=RuntimeError("disk-full")),
        ): bad = self.repo.log_training_metrics({"ts": datetime.utcnow()})
        self.assertFalse(bad["ok"])
        self.assertIsNotNone(bad["error"])
      # ok stays True, but payload is False and an error message is set
        self.assertFalse(bad["payload"])
        self.assertIsNotNone(bad["error"])


if __name__ == "__main__":             # pragma: no cover
    unittest.main(verbosity=2)
