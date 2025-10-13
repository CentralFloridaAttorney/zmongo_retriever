from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

from zmongo_toolbag.safe_result import SafeResult
from zmongo_toolbag.zmongo import ZMongo


class MongoOneHotDB:
    """
    Mongo-backed One-Hot 'words' dictionary using ZMongo (SafeResult-enabled).
    Compatible with the modern async ZMongo API.
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
        """Ensure indexes for dedupe/lookup."""
        if self._initialized:
            return
        if self._create_indexes:
            for field in ("word", "index"):
                try:
                    await self._zmongo.db[self._collection].create_index(field, unique=True)
                except Exception:
                    pass
        self._initialized = True

    # -------------------------- core ops --------------------------

    async def _get_next_index(self) -> int:
        """Compute next free index (0 if none)."""
        res = await self._zmongo.find(
            self._collection,
            {},
            sort=[("index", -1)],
            limit=1,
        )
        if not res.success or not res.data:
            return 0
        top = res.data[0]
        return int(top.get("index", -1)) + 1

    async def add_word(self, word: str) -> SafeResult:
        """Insert or update a word safely without MongoDB $set conflict errors."""
        await self.init()
        w = (word or "").strip()
        if not w:
            return SafeResult.fail("word is empty")

        existing = await self._zmongo.find_one(self._collection, {"word": w})
        if existing.success and existing.data:
            # Word exists — just update its timestamp
            updated = await self._zmongo.update_one(
                self._collection,
                {"word": w},
                {"$set": {"updated_at": datetime.now()}},
            )
            if not updated.success:
                return updated
            return existing

        # Insert new word with next available index
        next_idx = await self._get_next_index()
        doc = {
            "word": w,
            "index": next_idx,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        inserted = await self._zmongo.insert_one(self._collection, doc)
        if not inserted.success:
            return inserted

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
        sort = [("index", 1)] if sort_by_index else None
        r = await self._zmongo.find(self._collection, {}, sort=sort)
        if not r.success:
            return r
        return SafeResult.ok([row["word"] for row in (r.data or [])])

    async def size(self) -> SafeResult:
        await self.init()
        res = await self._zmongo.count_documents(self._collection, {})
        if not res.success:
            return res
        count_val = res.data.get("count", 0) if isinstance(res.data, dict) else res.data
        return SafeResult.ok(int(count_val))

    # -------------------------- vectorization helpers --------------------------

    async def ensure_words(self, tokens: Sequence[str]) -> SafeResult:
        await self.init()
        idxs: List[int] = []
        for t in tokens:
            put = await self.add_word(t)
            if not put.success:
                return put
            idxs.append(int(put.data["index"]))
        return SafeResult.ok(idxs)

    async def to_indices(self, tokens: Sequence[str]) -> SafeResult:
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
        await self.init()
        # delete_all_documents is sync — just call it directly
        return self._zmongo.delete_all_documents(self._collection)

    def close(self):
        self._zmongo.close()


# Demo
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
    db.close()


if __name__ == "__main__":
    asyncio.run(_demo())
