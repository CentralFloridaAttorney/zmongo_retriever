from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

try:
    # Preferred re-exports
    from zmongo_toolbag import ZMongo, SafeResult
except Exception:
    # Fallback to internal modules if re-exports aren’t present
    from zmongo_toolbag.zmongo import ZMongo  # type: ignore
    from zmongo_toolbag.data_processing import SafeResult  # type: ignore


class MongoOneHotDB:
    """
    A Mongo-backed One-Hot 'words' dictionary using ZMongo (SafeResult-enabled).

    Collection schema (default: "onehot_words"):
        {
          "_id": ObjectId,
          "word": str,        # unique
          "index": int,       # unique
          "created_at": datetime,
          "updated_at": datetime
        }

    API:
      - await add_word(word)            -> SafeResult(row)
      - await get_index(word)           -> SafeResult(int)
      - await get_word(index)           -> SafeResult(str)
      - await words(sort_by_index=True) -> SafeResult[List[str]]
      - await size()                    -> SafeResult[int]
      - await ensure_words(tokens)      -> SafeResult[List[int]]
      - await to_indices(tokens)        -> SafeResult[List[int]]
      - await to_one_hot_vector(word)   -> SafeResult[List[int]]
      - await to_bow_vector(tokens)     -> SafeResult[List[int]]
      - await delete_word(word)         -> SafeResult
      - await clear()                   -> SafeResult
    """

    def __init__(
        self,
        zmongo: Optional[ZMongo] = None,
        collection: str = "onehot_words",
        *,
        create_indexes: bool = True,
    ):
        self._zmongo = zmongo or ZMongo()
        self._collection = collection
        self._initialized = False
        self._create_indexes = create_indexes

    @property
    def collection(self) -> str:
        return self._collection

    async def init(self) -> None:
        """One-time initialization: create indexes for dedupe/lookup."""
        if self._initialized:
            return
        if self._create_indexes:
            try:
                await self._zmongo.create_index(self._collection, [("word", 1)], unique=True)
            except Exception:
                pass
            try:
                await self._zmongo.create_index(self._collection, [("index", 1)], unique=True)
            except Exception:
                pass
        self._initialized = True

    # -------------------------- core ops --------------------------

    async def _get_next_index(self) -> int:
        """Derive next free index = 0 if none, else max(index)+1."""
        res: SafeResult = await self._zmongo.find_documents(
            self._collection, {}, sort=[("index", -1)], limit=1
        )
        if not res.success or not res.data:
            return 0
        top = res.data[0]
        return int(top.get("index", -1)) + 1

    async def add_word(self, word: str) -> SafeResult:
        """Idempotently add a word. If it exists, return the existing row."""
        await self.init()
        w = (word or "").strip()
        if not w:
            return SafeResult.fail("word is empty")

        existing = await self._zmongo.find_one(self._collection, {"word": w})
        if existing.success and existing.data:
            return existing

        next_idx = await self._get_next_index()
        doc = {
            "word": w,
            "index": next_idx,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        up: SafeResult = await self._zmongo.update_one(
            self._collection,
            {"word": w},
            {"$setOnInsert": doc, "$set": {"updated_at": datetime.utcnow()}},
            upsert=True,
        )
        if not up.success:
            # race-safe fallback: read the row; if still missing, bubble error
            row = await self._zmongo.find_one(self._collection, {"word": w})
            return row if (row.success and row.data) else up

        # Return canonical row
        return await self._zmongo.find_one(self._collection, {"word": w})

    async def get_index(self, word: str) -> SafeResult:
        await self.init()
        r = await self._zmongo.find_one(self._collection, {"word": word})
        if not r.success or not r.data:
            return SafeResult.fail(f"word not found: {word}")
        return SafeResult.ok(int(r.data["index"]))

    async def get_word(self, index: int) -> SafeResult:
        await self.init()
        r = await self._zmongo.find_one(self._collection, {"index": int(index)})
        if not r.success or not r.data:
            return SafeResult.fail(f"index not found: {index}")
        return SafeResult.ok(str(r.data["word"]))

    async def words(self, *, sort_by_index: bool = True) -> SafeResult:
        await self.init()
        r = await self._zmongo.find_documents(
            self._collection, {}, sort=[("index", 1)] if sort_by_index else None
        )
        if not r.success:
            return r
        return SafeResult.ok([row["word"] for row in (r.data or [])])

    async def size(self) -> SafeResult:
        await self.init()
        return await self._zmongo.count_documents(self._collection, {})

    # -------------------------- vectorization helpers --------------------------

    async def ensure_words(self, tokens: Sequence[str]) -> SafeResult:
        """Ensure all tokens exist; returns their indices in order."""
        await self.init()
        idxs: List[int] = []
        for t in tokens:
            put = await self.add_word(t)
            if not put.success:
                return put
            idxs.append(int(put.data["index"]))
        return SafeResult.ok(idxs)

    async def to_indices(self, tokens: Sequence[str]) -> SafeResult:
        """Alias for ensure_words."""
        return await self.ensure_words(tokens)

    async def to_one_hot_vector(self, word: str) -> SafeResult:
        await self.init()
        idx_res = await self.get_index(word)
        if not idx_res.success:
            put = await self.add_word(word)
            if not put.success:
                return put
            idx = int(put.data["index"])
        else:
            idx = int(idx_res.data)

        size_res = await self.size()
        if not size_res.success:
            return size_res
        n = int(size_res.data)

        vec = [0] * n
        if 0 <= idx < n:
            vec[idx] = 1
        return SafeResult.ok(vec)

    async def to_bow_vector(self, tokens: Sequence[str]) -> SafeResult:
        await self.init()
        idxs_res = await self.to_indices(tokens)
        if not idxs_res.success:
            return idxs_res
        idxs = [int(i) for i in idxs_res.data]

        size_res = await self.size()
        if not size_res.success:
            return size_res
        n = int(size_res.data)

        vec = [0] * n
        for i in idxs:
            if 0 <= i < n:
                vec[i] += 1
        return SafeResult.ok(vec)

    # -------------------------- maintenance --------------------------

    async def delete_word(self, word: str) -> SafeResult:
        await self.init()
        return await self._zmongo.delete_one(self._collection, {"word": word})

    async def clear(self) -> SafeResult:
        """Dangerous: drops the collection (indexes will need re-created)."""
        await self.init()
        return await self._zmongo.drop_collection(self._collection)


# Optional local demo:
async def _demo():
    db = MongoOneHotDB()
    await db.init()
    await db.add_word("hello")
    await db.add_word("world")
    print("size:", (await db.size()).data)
    print("index('hello'):", (await db.get_index("hello")).data)
    print("word(1):", (await db.get_word(1)).data)
    print("words:", (await db.words()).data)
    print("one-hot('world'):", (await db.to_one_hot_vector("world")).data)
    print("bow(['hello','hello','world']):", (await db.to_bow_vector(['hello', 'hello', 'world'])).data)

if __name__ == "__main__":
    asyncio.run(_demo())
