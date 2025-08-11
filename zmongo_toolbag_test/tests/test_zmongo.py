import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from typing import List, AsyncGenerator

from dotenv import load_dotenv
from pymongo.operations import InsertOne

from zmongo_toolbag import ZMongo
from pydantic import BaseModel, Field

# --- Test Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")
TEST_DB_NAME = "zmongo_test_db"
os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME

# --- Pydantic Model for Testing ---
class Pet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    name: str
    age: int
    secret: str = Field(default="s", alias="_secret")

# --- Corrected Fixtures ---

@pytest_asyncio.fixture(scope="function")
async def zmongo() -> AsyncGenerator[ZMongo, None]:
    """
    Provide a ZMongo instance for each test function.

    This fixture now has "function" scope, creating a new client
    for each test and ensuring proper setup and teardown.
    """
    client = ZMongo()
    # The 'async with' block will automatically handle setup and
    # call client.__aexit__ for cleanup (closing the connection).
    async with client as zm:
        yield zm
        # After the test runs, drop the database to ensure a clean state.
        # This happens before the connection is closed by __aexit__.
        try:
            await zm.db.client.drop_database(TEST_DB_NAME)
        except Exception as e:
            print(f"Could not drop database {TEST_DB_NAME}: {e}")


@pytest.fixture(scope="function")
def coll_name() -> str:
    """
    Provide a unique collection name for each test function to prevent conflicts.
    """
    return f"col_{uuid.uuid4().hex}"


# --- Your Tests (Unchanged) ---

@pytest.mark.asyncio
async def test_insert_and_find_document(zmongo: ZMongo, coll_name: str):
    pet = Pet(name="Luna", age=3)
    res = await zmongo.insert_document(coll_name, pet)
    assert res.success and res.data.inserted_id  # type: ignore[attr-defined]

    found = await zmongo.find_document(coll_name, {"_id": res.data.inserted_id}) # type: ignore[attr-defined]
    assert found.success and found.data is not None
    assert found.data["name"] == "Luna"


@pytest.mark.asyncio
async def test_insert_many_and_find_many(zmongo: ZMongo, coll_name: str):
    pets: List[Pet] = [Pet(name=f"pet{i}", age=i) for i in range(5)]
    res = await zmongo.insert_documents(coll_name, pets)
    assert res.success and len(res.data.inserted_ids) == 5 # type: ignore[attr-defined]

    found = await zmongo.find_documents(coll_name, {"age": {"$gt": 2}})
    assert found.success and len(found.data) == 2


@pytest.mark.asyncio
async def test_update_document(zmongo: ZMongo, coll_name: str):
    pet = {"name": "Milo", "age": 4}
    ins = await zmongo.insert_document(coll_name, pet)
    assert ins.success

    upd = await zmongo.update_document(coll_name, {"_id": ins.data.inserted_id}, {"age": 5}) # type: ignore[attr-defined]
    assert upd.success and upd.data.modified_count == 1 # type: ignore[attr-defined]

    found = await zmongo.find_document(coll_name, {"_id": ins.data.inserted_id}) # type: ignore[attr-defined]
    assert found.success and found.data["age"] == 5


@pytest.mark.asyncio
async def test_update_many(zmongo: ZMongo, coll_name: str):
    await zmongo.insert_documents(
        coll_name, [{"cls": "A", "v": i} for i in range(3)] + [{"cls": "B", "v": 9}]
    )
    res = await zmongo.update_documents(coll_name, {"cls": "A"}, {"$set": {"v": -1}})
    assert res.success and res.data.modified_count == 3 # type: ignore[attr-defined]

    found = await zmongo.find_documents(coll_name, {"v": -1})
    assert found.success and len(found.data) == 3


@pytest.mark.asyncio
async def test_aggregate(zmongo: ZMongo, coll_name: str):
    await zmongo.insert_documents(
        coll_name,
        [{"kind": "x", "val": 1}, {"kind": "x", "val": 2}, {"kind": "y", "val": 9}],
    )
    pipeline = [{"$group": {"_id": "$kind", "total": {"$sum": "$val"}}}]
    res = await zmongo.aggregate(coll_name, pipeline)
    assert res.success
    results = {item["_id"]: item["total"] for item in res.data}
    assert results == {"x": 3, "y": 9}


@pytest.mark.asyncio
async def test_count_documents(zmongo: ZMongo, coll_name: str):
    await zmongo.insert_documents(coll_name, [{"a": 1}, {"a": 1}, {"a": 2}])
    res = await zmongo.count_documents(coll_name, {"a": 1})
    assert res.success and res.data["count"] == 2


@pytest.mark.asyncio
async def test_bulk_write_and_delete(zmongo: ZMongo, coll_name: str):
    ops = [InsertOne({"k": i}) for i in range(5)]
    bulk_res = await zmongo.bulk_write(coll_name, ops)
    assert bulk_res.success and bulk_res.data.inserted_count == 5 # type: ignore[attr-defined]

    del_res = await zmongo.delete_document(coll_name, {"k": 3})
    assert del_res.success and del_res.data.deleted_count == 1 # type: ignore[attr-defined]

    count_res = await zmongo.count_documents(coll_name, {})
    assert count_res.success and count_res.data["count"] == 4


@pytest.mark.asyncio
async def test_list_collections(zmongo: ZMongo, coll_name: str):
    await zmongo.insert_document(coll_name, {"foo": "bar"})
    res = await zmongo.list_collections()
    assert res.success and coll_name in res.data
