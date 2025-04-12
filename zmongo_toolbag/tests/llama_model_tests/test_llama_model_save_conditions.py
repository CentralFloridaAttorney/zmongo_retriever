import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from zmongo_toolbag.models.llama_model import LlamaModel

class TestLlamaModelSaveConditions(unittest.IsolatedAsyncioTestCase):

    @patch("zmongo_toolbag.models.llama_model.ZMongo")
    @patch("zmongo_toolbag.models.llama_model.Llama")
    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch.dict("os.environ", {
        "GGUF_MODEL_PATH": "/tmp/fake-model.bin"
    })
    async def test_save_conditions(self, mock_exists, mock_llama_class, mock_zmongo_class):
        # Create fake model
        mock_llama = MagicMock()
        mock_llama_class.return_value = mock_llama
        mock_llama.return_value = {"choices": [{"text": "Fake output"}]}

        # Create fake ZMongo with mocked update_document and close
        mock_zmongo = mock_zmongo_class.return_value
        mock_zmongo.update_document = AsyncMock(return_value=MagicMock(matched_count=1, upserted_id=None))
        mock_zmongo.mongo_client = MagicMock()

        model = LlamaModel(zmongo=None)  # this ensures _should_close is True

        # Condition 1: missing field name or text should raise ValueError
        with self.assertRaises(ValueError):
            await model.save("collection", ObjectId(), "", "text")

        with self.assertRaises(ValueError):
            await model.save("collection", ObjectId(), "field", "")

        # Condition 2: record_id as str should be converted to ObjectId
        string_id = str(ObjectId())
        await model.save("collection", string_id, "field", "generated", extra_fields={"a": "b"})
        args, kwargs = mock_zmongo.update_document.call_args
        self.assertEqual(kwargs["query"]["_id"], ObjectId(string_id))

        # Condition 3: If _should_close is True, mongo_client.close should be called
        self.assertTrue(model._should_close)
        mock_zmongo.mongo_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
