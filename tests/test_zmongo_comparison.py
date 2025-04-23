import pytest
import asyncio
import time
from zmongo_toolbag_dev.zmongo import ZMongo as ZMongoMotor
from zmongo_toolbag.zmongo import ZMongo as ZMongoLegacy  # Path to pymongo-based version

TEST_COLLECTION = "compare_zmongo"

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_compare_all_operations():
    motor = ZMongoMotor()
    legacy = ZMongoLegacy()

    doc = {"name": "SpeedTest", "value": 42}

    # Insert
    start = time.perf_counter()
    inserted_motor = await motor.insert_document(TEST_COLLECTION, doc.copy())
    insert_motor_time = time.perf_counter() - start

    start = time.perf_counter()
    inserted_legacy = await legacy.insert_document(TEST_COLLECTION, doc.copy())
    insert_legacy_time = time.perf_counter() - start

    query = {"_id": inserted_motor["_id"]}

    # Find
    start = time.perf_counter()
    await motor.find_document(TEST_COLLECTION, query)
    find_motor_time = time.perf_counter() - start

    start = time.perf_counter()
    await legacy.find_document(TEST_COLLECTION, query)
    find_legacy_time = time.perf_counter() - start

    # Update
    update_data = {"$set": {"value": 99}}
    start = time.perf_counter()
    await motor.update_document(TEST_COLLECTION, query, update_data)
    update_motor_time = time.perf_counter() - start

    start = time.perf_counter()
    await legacy.update_document(TEST_COLLECTION, query, update_data)
    update_legacy_time = time.perf_counter() - start

    # Delete
    start = time.perf_counter()
    await motor.delete_document(TEST_COLLECTION, query)
    delete_motor_time = time.perf_counter() - start

    start = time.perf_counter()
    await legacy.delete_document(TEST_COLLECTION, query)
    delete_legacy_time = time.perf_counter() - start

    # Print Results
    print(f"\nInsert - Motor: {insert_motor_time:.6f}s | Legacy: {insert_legacy_time:.6f}s")
    print(f"Find   - Motor: {find_motor_time:.6f}s | Legacy: {find_legacy_time:.6f}s")
    print(f"Update - Motor: {update_motor_time:.6f}s | Legacy: {update_legacy_time:.6f}s")
    print(f"Delete - Motor: {delete_motor_time:.6f}s | Legacy: {delete_legacy_time:.6f}s")

    # Assert all completed
    assert True
