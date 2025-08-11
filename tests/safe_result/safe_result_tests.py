import unittest
from bson.objectid import ObjectId
from datetime import datetime
import pandas as pd
import numpy as np

from safe_result import SafeResult


# Assume SafeResult and DataProcessor are in the same module as above
# from mymodule import SafeResult, DataProcessor

class TestSafeResult(unittest.TestCase):
    def test_basic_serialization_dict(self):
        doc = {"_id": ObjectId("5f50c31e7b1e8a9459b8b73a"), "x": 1, "dt": datetime(2024, 1, 2, 3, 4, 5)}
        result = SafeResult(doc)
        dump = result.model_dump()
        self.assertEqual(dump["_id"], "5f50c31e7b1e8a9459b8b73a")
        self.assertEqual(dump["x"], 1)
        self.assertTrue(isinstance(dump["dt"], str) and dump["dt"].startswith("2024-01-02T03:04:05"))

    def test_serialization_list(self):
        data = [ObjectId("5f50c31e7b1e8a9459b8b73a"), {"y": 2}]
        result = SafeResult(data)
        dump = result.model_dump()
        self.assertEqual(dump[0], "5f50c31e7b1e8a9459b8b73a")
        self.assertEqual(dump[1]["y"], 2)

    def test_flatten_and_keys(self):
        data = {"foo": {"bar": [1, {"baz": 2}]}}
        result = SafeResult(data)
        flat = result.flatten()
        self.assertEqual(flat["foo.bar.0"], 1)
        self.assertEqual(flat["foo.bar.1.baz"], 2)
        keys = result.get_keys()
        self.assertIn("foo.bar.1.baz", keys)

    def test_get_value(self):
        data = {"foo": {"bar": [{"baz": 5}, {"baz": 10}]}}
        result = SafeResult(data)
        self.assertEqual(result.get_value("foo.bar.1.baz"), 10)
        self.assertIsNone(result.get_value("foo.missing"))
        self.assertIsNone(result.get_value("foo.bar.2.baz"))

    def test_contains_and_getitem(self):
        data = {"a": 123, "b": 456}
        result = SafeResult(data)
        self.assertIn("a", result)
        self.assertEqual(result["b"], 456)
        with self.assertRaises(TypeError):
            SafeResult([1, 2, 3])["a"]

    def test_is_empty_and_bool(self):
        self.assertTrue(SafeResult({}).is_empty())
        self.assertTrue(SafeResult([]).is_empty())
        self.assertFalse(SafeResult({"foo": 1}).is_empty())
        self.assertTrue(not SafeResult({}))
        self.assertTrue(SafeResult({"foo": 1}))

    def test_repr(self):
        data = {"hello": "world"}
        result = SafeResult(data)
        r = repr(result)
        self.assertIn("hello", r)
        self.assertIn("SafeResult", r)

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = SafeResult(df)
        dump = result.model_dump()
        self.assertTrue(isinstance(dump, list))
        self.assertEqual(dump[0]["a"], 1)
        self.assertEqual(dump[1]["b"], 4)

    def test_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = SafeResult(arr)
        dump = result.model_dump()
        self.assertEqual(dump, [[1, 2], [3, 4]])

    def test_bytes(self):
        b = b"hello"
        result = SafeResult(b)
        dump = result.model_dump()
        self.assertEqual(dump, "hello")

    def test_deep_circular_reference(self):
        a = {}
        a["self"] = a
        result = SafeResult(a)
        dump = result.model_dump()
        # Should not recurse infinitely
        self.assertIn("self", dump)
        self.assertIn("__circular_reference__", str(dump["self"]))

    def test_metadata(self):
        doc = {"a": {"b": {"c": 123}}, "x": [1, 2, {"y": 5}]}
        result = SafeResult(doc)
        meta = result.get_metadata()
        # Should have dot notation keys
        self.assertIn("a.b.c", meta)
        self.assertIn("x.2.y", meta)
        self.assertEqual(meta["a.b.c"], 123)
        self.assertEqual(meta["x.2.y"], 5)

if __name__ == "__main__":
    unittest.main()
