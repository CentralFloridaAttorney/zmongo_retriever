import unittest
from bson import ObjectId

from zmongo_toolbag.utils.data_processing import DataProcessing


class TestGetOidValue(unittest.TestCase):

    def test_valid_objectid(self):
        oid = ObjectId()
        result = DataProcessing.get_oid_value(oid)
        self.assertEqual(result, oid)

    def test_dict_with_oid(self):
        oid = ObjectId()
        result = DataProcessing.get_oid_value({'$oid': str(oid)})
        self.assertEqual(result, oid)

    def test_stringified_dict_with_oid(self):
        oid = ObjectId()
        bson_str = str({'$oid': str(oid)})  # e.g., "{'$oid': '...'}"
        result = DataProcessing.get_oid_value(bson_str)
        self.assertEqual(result, oid)

    def test_plain_valid_oid_string(self):
        oid = ObjectId()
        result = DataProcessing.get_oid_value(str(oid))
        self.assertEqual(result, oid)

    def test_invalid_oid_string(self):
        result = DataProcessing.get_oid_value("not_a_valid_oid")
        self.assertIsNone(result)

    def test_dict_without_oid_key(self):
        result = DataProcessing.get_oid_value({'not_oid': 'value'})
        self.assertIsNone(result)

    def test_stringified_invalid_dict(self):
        bad_str = "{'not_oid': 'value'}"
        result = DataProcessing.get_oid_value(bad_str)
        self.assertIsNone(result)

    def test_non_string_non_dict_non_oid_input(self):
        result = DataProcessing.get_oid_value(12345)
        self.assertIsNone(result)

    def test_none_input(self):
        result = DataProcessing.get_oid_value(None)
        self.assertIsNone(result)

    def test_malformed_string_eval(self):
        # This would raise a SyntaxError in ast.literal_eval
        bad_str = "{'$oid': 'missing_quote}"
        result = DataProcessing.get_oid_value(bad_str)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
