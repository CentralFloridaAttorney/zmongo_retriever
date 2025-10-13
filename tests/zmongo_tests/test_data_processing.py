import unittest
import re
from datetime import datetime
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
from collections import deque
import html

from zmongo_toolbag.data_processing import DataProcessor


# Assuming the DataProcessor class is in a file named data_processing.py


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.nested_obj = {
            "a": 1,
            "b": {
                "c": "hello",
                "d": [10, 20, {"e": 30}]
            },
            "f": None
        }

    def test_get_value(self):
        self.assertEqual(DataProcessor.get_value(self.nested_obj, "a"), 1)
        self.assertEqual(DataProcessor.get_value(self.nested_obj, "b.c"), "hello")
        self.assertEqual(DataProcessor.get_value(self.nested_obj, "b.d.1"), 20)
        self.assertEqual(DataProcessor.get_value(self.nested_obj, "b.d.2.e"), 30)
        self.assertIsNone(DataProcessor.get_value(self.nested_obj, "b.d.3"))
        self.assertIsNone(DataProcessor.get_value(self.nested_obj, "x.y.z"))
        self.assertIsNone(DataProcessor.get_value(self.nested_obj, "f"))

    def test_set_value(self):
        # Test setting existing value in dict
        self.assertTrue(DataProcessor.set_value(self.nested_obj, "b.c", "world"))
        self.assertEqual(self.nested_obj["b"]["c"], "world")

        # Test setting existing value in list
        self.assertTrue(DataProcessor.set_value(self.nested_obj, "b.d.0", 100))
        self.assertEqual(self.nested_obj["b"]["d"][0], 100)

        # Test creating new path in dict
        self.assertTrue(DataProcessor.set_value(self.nested_obj, "b.f.g", 50))
        self.assertEqual(self.nested_obj["b"]["f"]["g"], 50)

        # Test failure on out-of-bounds list index
        self.assertFalse(DataProcessor.set_value(self.nested_obj, "b.d.5", 500))

        # Test setting a value to None
        self.assertTrue(DataProcessor.set_value(self.nested_obj, "a", None))
        self.assertIsNone(self.nested_obj["a"])

    def test_flatten_json(self):
        flat_dict = DataProcessor.flatten_json(self.nested_obj)
        expected = {
            "a": 1,
            "b.c": "hello",
            "b.d.0": 10,
            "b.d.1": 20,
            "b.d.2.e": 30,
            "f": None
        }
        self.assertEqual(flat_dict, expected)

        # Test with a standalone value
        self.assertEqual(DataProcessor.flatten_json(123), {})
        self.assertEqual(DataProcessor.flatten_json(123, "key"), {"key": 123})

    def test_clean_output_text(self):
        text1 = "```html\n<p>Hello</p>\n```"
        self.assertEqual(DataProcessor.clean_output_text(text1), "<p>Hello</p>")
        text2 = "Just some text"
        self.assertEqual(DataProcessor.clean_output_text(text2), "Just some text")
        text3 = "```\nContent\n```"
        self.assertEqual(DataProcessor.clean_output_text(text3), "Content")
        with self.assertRaises(ValueError):
            DataProcessor.clean_output_text(123)

    def test_convert_object_to_json(self):
        class SimpleObject:
            def __init__(self):
                self.public_attr = "visible"
                self._private_attr = "hidden"

        class CircularRef:
            pass

        obj1 = CircularRef()
        obj2 = CircularRef()
        obj1.ref = obj2
        obj2.ref = obj1

        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        series = pd.Series([5, 6, 7])

        data_to_convert = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "objectid": ObjectId("615f7b4b7c3b2e2a1b7d8c3c"),
            "dataframe": df,
            "series": series,
            "numpy_array": np.array([1, 2, 3]),
            "custom_object": SimpleObject(),
            "a_set": {1, 2, 3},
            "a_deque": deque([4, 5, 6]),
            "circular": obj1,
            "bytes": b"hello"
        }

        converted = DataProcessor.convert_object_to_json(data_to_convert)

        self.assertEqual(converted["datetime"], "2023-01-01T12:00:00")
        self.assertEqual(converted["objectid"], "615f7b4b7c3b2e2a1b7d8c3c")
        self.assertEqual(converted["dataframe"], [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}])
        self.assertEqual(converted["series"], {0: 5, 1: 6, 2: 7})
        self.assertEqual(converted["numpy_array"], [1, 2, 3])
        self.assertEqual(converted["custom_object"], {'public_attr': 'visible'})
        self.assertIsInstance(converted["a_set"], list)
        self.assertIsInstance(converted["a_deque"], list)
        self.assertEqual(converted["circular"], {'ref': {'ref': {'__circular_reference__': 'CircularRef'}}})
        self.assertEqual(converted["bytes"], "hello")

    def test_convert_text_to_html(self):
        # Test with simple HTML string
        html_str = "&lt;p&gt;This is a test with &amp; ampersand.&lt;/p&gt;"
        result_html = DataProcessor.convert_text_to_html(html_str)
        # Normalize whitespace for consistent comparison
        normalized_result = ' '.join(result_html.split())
        self.assertIn("<p>This is a test with & ampersand.</p>", normalized_result)

        # Test with dictionary input
        data_dict = {"output_text": "<h1>Title &amp; Stuff</h1>"}
        result_dict_html = DataProcessor.convert_text_to_html(data_dict)
        normalized_dict_result = ' '.join(result_dict_html.split())
        self.assertIn("<h1>Title & Stuff</h1>", normalized_dict_result)

        # Test with invalid input types
        with self.assertRaises(ValueError):
            DataProcessor.convert_text_to_html(12345)
        with self.assertRaises(ValueError):
            DataProcessor.convert_text_to_html({"wrong_key": "value"})


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

