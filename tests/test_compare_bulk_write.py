import pytest
import asyncio
import time
from zmongo_toolbag_dev.zmongo import ZMongo as ZMongoMotor
from zmongo_toolbag.zmongo import ZMongo as ZMongoLegacy

TEST_COLLECTION = "bulk_write_test"

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_bulk_write_comparison():
    motor = ZMongoMotor()
    legacy = ZMongoLegacy()

    # Prepare operations
    ops_motor = [
        {"operation": "insert", "document": {"name": f"motor_{i}"}} for i in range(100)
    ] + [
        {"operation": "update", "filter": {"name": f"motor_{i}"}, "update": {"$set": {"updated": True}}} for i in range(50)
    ] + [
        {"operation": "delete", "filter": {"name": f"motor_{i}"}} for i in range(25)
    ]

    # Motor timing
    start_motor = time.perf_counter()
    result_motor = await motor.bulk_write(TEST_COLLECTION, ops_motor)
    duration_motor = time.perf_counter() - start_motor

    # Prepare legacy-specific operations
    from pymongo import InsertOne, UpdateOne, DeleteOne
    ops_legacy = [
        InsertOne({"name": f"motor_{i}"}) for i in range(100)
    ] + [
        UpdateOne({"name": f"motor_{i}"}, {"$set": {"updated": True}}) for i in range(50)
    ] + [
        DeleteOne({"name": f"motor_{i}"}) for i in range(25)
    ]

    # Legacy timing
    start_legacy = time.perf_counter()
    result_legacy = await legacy.bulk_write(TEST_COLLECTION, ops_legacy)
    duration_legacy = time.perf_counter() - start_legacy

    # Print results
    print("\nMotor bulk_write result:", result_motor)
    print("Motor duration: {:.6f}s".format(duration_motor))
    print("Legacy bulk_write result:", result_legacy)
    print("Legacy duration: {:.6f}s".format(duration_legacy))

    assert result_motor.get("acknowledged") is True
    assert result_legacy.get("acknowledged") is True
