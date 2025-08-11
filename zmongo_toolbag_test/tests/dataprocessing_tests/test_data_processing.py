import unittest
from bson.objectid import ObjectId
from datetime import datetime
import pandas as pd
import numpy as np

from zmongo_toolbag.utils.data_processing import DataProcessor


class DummyWithToDict:
    def to_dict(self):
        raise Exception("Intentional failure")


class DummyUnserializable:
    __slots__ = ()
    def __str__(self):
        return "fallback"


class DummyObject:
    def __init__(self):
        self.value = 42


class DummyWithAttrs:
    def __init__(self):
        self.foo = "bar"
        self.baz = 123
        self._private = "hidden"
        self.method = lambda: None


class DummyWithExtras:
    def __init__(self):
        self.keep = "yes"
        self._skip = "no"
        self.func = lambda: "callable"


class PrivateOnly:
    def __init__(self):
        self._hidden = "secret"
    def _method(self):
        return "nope"


class DeepNest:
    def __init__(self, level=0):
        if level <= 101:
            self.child = DeepNest(level + 1)


class TestDataProcessor(unittest.TestCase):

    # 1. Text Cleaning
    def test_clean_output_text(self):
        raw = "```html\n<p>Hello!</p>\n```"
        expected = "<p>Hello!</p>"
        self.assertEqual(DataProcessor.clean_output_text(raw), expected)

    def test_clean_output_text_non_string(self):
        with self.assertRaises(ValueError):
            DataProcessor.clean_output_text(123)

    # 2. Object to JSON Conversion
    def test_convert_object_to_json_basic(self):
        obj = {
            "_id": ObjectId("65f1b6beae7cd4d4d1d3ae8d"),
            "created": datetime(2024, 1, 1),
            "list": [1, 2, 3],
            "data": {"nested": "value"},
        }
        result = DataProcessor.convert_object_to_json(obj)
        self.assertEqual(result["_id"], str(obj["_id"]))
        self.assertEqual(result["created"], obj["created"].isoformat())

    def test_convert_object_to_json_bytes(self):
        result = DataProcessor.convert_object_to_json(b"hello")
        self.assertEqual(result, "hello")

    def test_convert_object_to_json_circular(self):
        obj = []
        obj.append(obj)
        result = DataProcessor.convert_object_to_json(obj)
        self.assertEqual(result[0], {"__circular_reference__": "list"})

    def test_convert_object_to_json_to_dict_error(self):
        obj = DummyWithToDict()
        result = DataProcessor.convert_object_to_json(obj)
        self.assertEqual(result, str(obj))

    def test_convert_object_to_json_dunder_dict(self):
        obj = DummyWithAttrs()
        result = DataProcessor.convert_object_to_json(obj)
        self.assertIn("foo", result)
        self.assertIn("baz", result)
        self.assertNotIn("_private", result)
        self.assertNotIn("method", result)

    def test_convert_object_to_json_dataframe(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        result = DataProcessor.convert_object_to_json(df)
        expected = df.to_dict(orient="records")
        self.assertEqual(result, expected)

    def test_convert_object_to_json_filters_dunder_and_callable(self):
        obj = DummyWithExtras()
        result = DataProcessor.convert_object_to_json(obj)
        self.assertEqual(result, {"keep": "yes"})

    def test_convert_object_to_json_series(self):
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        result = DataProcessor.convert_object_to_json(series)
        expected = series.to_dict()
        self.assertEqual(result, expected)

    def test_convert_object_to_json_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        result = DataProcessor.convert_object_to_json(arr)
        expected = arr.tolist()
        self.assertEqual(result, expected)

    def test_convert_object_to_json_depth_limit(self):
        class Recursive:
            def __init__(self):
                self.ref = self

        obj = Recursive()
        result = DataProcessor.convert_object_to_json(obj)
        self.assertIn("ref", result)
        self.assertEqual(result["ref"], {"__circular_reference__": "Recursive"})

    def test_convert_object_to_json_max_depth_exceeded(self):
        deep_obj = DeepNest()
        result = DataProcessor.convert_object_to_json(deep_obj)

        current = result
        for _ in range(200):
            if isinstance(current, dict) and "child" in current:
                current = current["child"]
            else:
                break

        self.assertIn("__error__", current)
        self.assertEqual(current["__error__"], "Maximum depth exceeded")

    def test_convert_object_to_json_str_fallback_on_dict_only_private(self):
        obj = PrivateOnly()
        result = DataProcessor.convert_object_to_json(obj)
        self.assertEqual(result, str(obj))

    # 3. JSON Metadata Handling
    def test_convert_json_to_metadata(self):
        data = {"a": {"b": 1}, "c": [2, 3]}
        result = DataProcessor.convert_json_to_metadata(data)
        self.assertEqual(result["a.b"], 1)
        self.assertEqual(result["c.0"], 2)

    def test_convert_mongo_to_metadata(self):
        doc = {"user": {"name": "Alice"}, "tags": ["x", "y"]}
        result = DataProcessor.convert_mongo_to_metadata(doc)
        self.assertEqual(result["user_name"], "Alice")
        self.assertEqual(result["tags_0"], "x")

    def test_convert_mongo_to_metadata_none(self):
        result = DataProcessor.convert_mongo_to_metadata(None)
        self.assertEqual(result, {})

    def test_convert_mongo_to_metadata_list_of_dicts(self):
        data = {
            "items": [
                {"a": 1},
                {"b": 2},
                [3, 4]
            ]
        }
        result = DataProcessor.convert_mongo_to_metadata(data)
        expected = {
            "items_0_a": 1,
            "items_1_b": 2,
            "items_2_0": 3,
            "items_2_1": 4
        }
        self.assertEqual(result, expected)

    # 4. HTML Formatting
    def test_convert_text_to_html_string(self):
        text = "<p>Hello &amp; Welcome</p>"
        html_out = DataProcessor.convert_text_to_html(text)
        self.assertIn("&amp;", html_out)

    def test_convert_text_to_html_invalid_dict(self):
        with self.assertRaises(ValueError):
            DataProcessor.convert_text_to_html({"wrong_key": "value"})

    def test_convert_text_to_html_invalid_type(self):
        with self.assertRaises(ValueError):
            DataProcessor.convert_text_to_html(42)

    # 5. JSON Structure Navigation
    def test_get_value(self):
        doc = {"a": {"b": {"c": 123}}}
        self.assertEqual(DataProcessor.get_value(doc, "a.b.c"), 123)
        self.assertIsNone(DataProcessor.get_value(doc, "a.b.x"))

    def test_get_value_with_list_index(self):
        data = {"items": ["zero", "one", "two"]}
        result = DataProcessor.get_value(data, "items.1")
        self.assertEqual(result, "one")

    def test_get_value_with_invalid_list_index(self):
        data = {"items": ["zero", "one"]}
        result = DataProcessor.get_value(data, "items.foo")
        self.assertIsNone(result)

    def test_flatten_json(self):
        data = {"a": {"b": 1}, "c": [2, 3]}
        flat = DataProcessor.flatten_json(data)
        self.assertEqual(flat["a.b"], 1)
        self.assertEqual(flat["c.0"], 2)
        self.assertEqual(flat["c.1"], 3)

    def test_get_keys_from_json(self):
        data = {"a": {"b": 1}, "c": [2, 3]}
        keys = DataProcessor.get_keys_from_json(data)
        self.assertIn("a.b", keys)
        self.assertIn("c.0", keys)

    # 6. Embedding / Vector Handling
    def test_get_values_as_list(self):
        df = pd.DataFrame({
            "embedding_0": [0.1, 0.2],
            "embedding_1": [0.3, 0.4],
            "other": ["x", "y"]
        })
        result = DataProcessor.get_values_as_list(df)
        self.assertEqual(result, [0.1, 0.3, 0.2, 0.4])

    # 7. Serialization
    def test_convert_object_to_serializable_list_of_objects(self):
        data = [
            ObjectId("65fc1ec8dd5dfcff0bde5c5e"),
            datetime(2024, 1, 1),
            {"nested": ObjectId("65fc1ec8dd5dfcff0bde5c5f")},
        ]
        result = DataProcessor.convert_object_to_serializable(data)
        self.assertEqual(result[0], "65fc1ec8dd5dfcff0bde5c5e")
        self.assertEqual(result[1], "2024-01-01T00:00:00")
        self.assertEqual(result[2], {"nested": "65fc1ec8dd5dfcff0bde5c5f"})

    def test_convert_object_to_serializable(self):
        obj = {
            "id": ObjectId(),
            "time": datetime(2020, 5, 1),
            "df": pd.DataFrame({"a": [1, 2]}),
            "arr": np.array([1, 2]),
        }
        result = DataProcessor.convert_object_to_serializable(obj)
        self.assertEqual(result["arr"], [1, 2])

    def test_convert_object_to_serializable_fallback(self):
        obj = DummyUnserializable()
        result = DataProcessor.convert_object_to_serializable(obj)
        self.assertEqual(result, "fallback")

    def test_convert_object_to_serializable_with_dunder_dict(self):
        dummy = DummyObject()
        result = DataProcessor.convert_object_to_serializable(dummy)
        self.assertIn("value", result)

    # 8. Unicode Detection
    def test_detect_unicode_surrogates(self):
        self.assertTrue(DataProcessor.detect_unicode_surrogates("\uD83D\uDE00"))
        self.assertFalse(DataProcessor.detect_unicode_surrogates("plain"))

    def test_detect_unicode_surrogates_invalid_type(self):
        with self.assertRaises(ValueError):
            DataProcessor.detect_unicode_surrogates(123)

    # 9. Opinion Extraction
    def test_get_opinion_from_zcase(self):
        zcase = {"casebody": {"data": {"opinions": [{"text": "This is the opinion."}]}}}
        self.assertEqual(DataProcessor.get_opinion_from_zcase(zcase), "This is the opinion.")

    def test_get_opinion_from_zcase_no_opinions(self):
        zcase = {"casebody": {"data": {"opinions": []}}}
        result = DataProcessor.get_opinion_from_zcase(zcase)
        self.assertEqual(result, "No opinions found.")

    def test_get_opinion_from_zcase_exception(self):
        zcase = {"casebody": None}
        result = DataProcessor.get_opinion_from_zcase(zcase)
        self.assertTrue(result.startswith("Error extracting opinion"))

    # 10. Utility
    def test_list_to_string(self):
        items = [1, "a", 2]
        self.assertEqual(DataProcessor.list_to_string(items), "1a2")


if __name__ == "__main__":
    unittest.main()