import os
import pytest
from bson import ObjectId
from zmongo_toolbag.mongo_onehot_db import MongoOneHotDB
from zmongo_toolbag.zmongo import ZMongo

def unwrap(result):
    assert result.success, f"Operation failed: {result.error}"
    return result.data

@pytest.fixture(scope="function")
def onehot_db():
    """Provides a clean MongoOneHotDB instance for sync ZMongo."""
    db_name = f"test_onehot_{ObjectId()}"
    os.environ["MONGO_DATABASE_NAME"] = db_name
    zm = ZMongo()
    ohdb = MongoOneHotDB(zmongo=zm)
    ohdb.init_sync = True
    ohdb._collection = "test_onehot_words"
    yield ohdb
    zm.delete_all_documents(ohdb._collection)
    zm.drop_database(db_name)

def test_add_and_get_index(onehot_db):
    res1 = onehot_db.add_word_sync("alpha")
    assert res1.success
    idx = unwrap(res1)
    assert isinstance(idx, int)

def test_get_word_and_index(onehot_db):
    onehot_db.add_word_sync("beta")
    idx = unwrap(onehot_db.get_index("beta"))
    word = unwrap(onehot_db.get_word(idx))
    assert word == "beta"

def test_words_and_size(onehot_db):
    onehot_db.add_word_sync("cat")
    onehot_db.add_word_sync("dog")
    all_words = unwrap(onehot_db.words())
    size = unwrap(onehot_db.size())
    assert "cat" in all_words and "dog" in all_words
    assert size >= 2

def test_to_one_hot_vector(onehot_db):
    onehot_db.add_word_sync("x")
    onehot_db.add_word_sync("y")
    v = unwrap(onehot_db.to_one_hot_vector("x"))
    assert sum(v) == 1

def test_to_bow_vector(onehot_db):
    onehot_db.ensure_words_sync(["red", "blue", "red"])
    bow = unwrap(onehot_db.to_bow_vector(["red", "blue", "red"]))
    assert bow.sum() >= 3

def test_clear_and_re_add(onehot_db):
    onehot_db.add_word_sync("zeta")
    before = unwrap(onehot_db.size())
    assert before >= 1
    clear_res = onehot_db.clear()
    assert clear_res.success
    onehot_db.add_word_sync("eta")
    after = unwrap(onehot_db.size())
    assert after >= 1
