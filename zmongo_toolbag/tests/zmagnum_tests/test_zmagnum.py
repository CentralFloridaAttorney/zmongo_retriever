import asyncio
import os
import sys
import pytest
import pytest_asyncio
from bson import ObjectId

# Add project root to sys.path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from zmongo_toolbag.zmagnum import ZMagnum, UpdateResponse

@pytest_asyncio.fixture(scope="function")
async def zmagnum_instance():
    os.environ["MONGO_URI"] = "mongodb://localhost:27017"
    os.environ["MONGO_DATABASE_NAME"] = "test_zmagnum_db"
    zmagnum = ZMagnum(disable_cache=False)
    await zmagnum.db["test_collection"].delete_many({})
    yield zmagnum
    await zmagnum.db["test_collection"].delete_many({})
    await zmagnum.close()

@pytest.mark.asyncio
async def test_insert_and_find_document(zmagnum_instance):
    doc = {"name": "test_insert", "value": 42}
    result = await zmagnum_instance.insert_documents("test_collection", [doc])
    assert result["inserted_count"] == 1

    found = await zmagnum_instance.find_document("test_collection", {"name": "test_insert"})
    assert found is not None
    assert found["value"] == 42

@pytest.mark.asyncio
async def test_update_document(zmagnum_instance):
    doc = {"name": "test_update", "value": 1}
    await zmagnum_instance.insert_documents("test_collection", [doc])

    result: UpdateResponse = await zmagnum_instance.update_document(
        "test_collection", {"name": "test_update"}, {"$set": {"value": 99}}
    )
    assert result.matched_count == 1
    assert result.modified_count == 1

    updated = await zmagnum_instance.find_document("test_collection", {"name": "test_update"})
    assert updated["value"] == 99

@pytest.mark.asyncio
async def test_recommend_indexes(zmagnum_instance):
    docs = [{"field1": i, "field2": i * 2} for i in range(50)]
    await zmagnum_instance.insert_documents("test_collection", docs)
    recommendations = await zmagnum_instance.recommend_indexes("test_collection", sample_size=50)
    assert "field1" in recommendations
    assert "field2" in recommendations

@pytest.mark.asyncio
async def test_embedding_schema_analysis(zmagnum_instance):
    docs = [
        {"embedding": [float(i) for i in range(128)]} for _ in range(20)
    ]
    await zmagnum_instance.insert_documents("test_collection", docs)
    analysis = await zmagnum_instance.analyze_embedding_schema("test_collection")
    assert analysis["sampled"] == 20
    assert analysis["avg_embedding_length"] == 128

@pytest.mark.asyncio
async def test_is_sharded_cluster(zmagnum_instance):
    result = zmagnum_instance.is_sharded_cluster()
    assert isinstance(result, bool)

@pytest.mark.asyncio
async def test_route_to_shard(zmagnum_instance):
    shard = zmagnum_instance.route_to_shard("somekey")
    assert shard.startswith("shard-")