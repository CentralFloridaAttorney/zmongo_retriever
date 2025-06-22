from bson import ObjectId

from zmongo_toolbag.utils.safe_result import SafeResult


def test_ok_and_fail_basic():
    # ok
    s = SafeResult.ok({"x": 1})
    assert s.success
    assert s.error is None
    assert s.data == {"x": 1}
    # fail
    f = SafeResult.fail("some error", {"y": 2})
    assert not f.success
    assert f.error == "some error"
    assert f.data == {"y": 2}

def test_bson_objectid_conversion():
    oid = ObjectId()
    s = SafeResult.ok({"_id": oid, "foo": "bar"})
    # .data should have stringified _id
    assert isinstance(s.data["_id"], str)
    assert s.data["foo"] == "bar"
    # .original should restore ObjectId
    restored = s.original()
    assert isinstance(restored["_id"], ObjectId)
    assert restored["foo"] == "bar"

def test_keymap_restoration():
    fake_doc = {
        "name": "Whiskers",
        "usecret": "purr",
        "_id": str(ObjectId()),
        "__keymap": {"usecret": "_secret"}
    }
    s = SafeResult.ok(fake_doc)
    orig = s.original()
    assert "_secret" in orig
    assert orig["_secret"] == "purr"
    assert "usecret" not in orig

def test_list_of_docs_with_keymap():
    oid = ObjectId()
    docs = [
        {
            "name": "A",
            "usecret": "x",
            "_id": str(oid),
            "__keymap": {"usecret": "_secret"}
        },
        {
            "name": "B",
            "usecret": "y",
            "_id": str(oid),
            "__keymap": {"usecret": "_secret"}
        },
    ]
    s = SafeResult.ok(docs)
    orig = s.original()
    assert isinstance(orig, list)
    for d in orig:
        assert "_secret" in d
        assert "usecret" not in d

def test_model_dump_and_to_json():
    s = SafeResult.ok({"foo": "bar"})
    d = s.model_dump()
    assert d["success"] is True
    assert d["data"] == {"foo": "bar"}
    assert d["error"] is None
    # JSON is valid and contains keys
    j = s.to_json()
    assert '"success": true' in j
    assert '"foo": "bar"' in j

def test_fail_with_no_data():
    s = SafeResult.fail("fail msg")
    assert not s.success
    assert s.error == "fail msg"
    assert s.data is None

def test_repr():
    s = SafeResult.ok({"foo": "bar"})
    r = repr(s)
    assert "SafeResult(success=True" in r
    assert "foo" in r



def test_original_handles_non_objectid_string():
    # Provide a dict with a string _id that is NOT a valid ObjectId
    d = {"_id": "notanobjectid", "foo": "bar"}
    sr = SafeResult.ok(d)
    # original() should not raise, and _id should remain a string
    orig = sr.original()
    assert isinstance(orig, dict)
    assert orig["_id"] == "notanobjectid"


def test_original_return_data_fallback():
    # Pass in an int (primitive), triggers the final "return data" line
    sr = SafeResult.ok(12345)
    assert sr.original() == 12345

    # Pass in a float
    sr = SafeResult.ok(3.1415)
    assert sr.original() == 3.1415

    # Pass in a string
    sr = SafeResult.ok("just a string")
    assert sr.original() == "just a string"

    # Pass in None
    sr = SafeResult.ok(None)
    assert sr.original() is None

    # Pass in a boolean
    sr = SafeResult.ok(False)
    assert sr.original() is False
