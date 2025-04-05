# test_zmongo_embedder.py

import os
import pytest
import hashlib
from bson.objectid import ObjectId
from unittest.mock import MagicMock, AsyncMock, patch, call
import pytest_asyncio

# Update the import to match your real path:
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo import ZMongo


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Fixture to set environment variables used by ZMongoEmbedder.
    """
    monkeypatch.setenv("OPENAI_API_KEY_APP", "test_api_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("EMBEDDING_TOKEN_LIMIT", "8192")
    monkeypatch.setenv("EMBEDDING_ENCODING", "cl100k_base")


@pytest_asyncio.fixture
async def mock_repository():
    """
    Fixture to create a mock of the ZMongo repository.
    """
    repo = MagicMock(spec=ZMongo)
    repo.find_document = AsyncMock()
    repo.insert_document = AsyncMock()
    repo.save_embedding = AsyncMock()
    return repo


@pytest_asyncio.fixture
async def embedder(mock_env_vars, mock_repository):
    """
    Returns an instance of ZMongoEmbedder with the mocked repository.
    """
    return ZMongoEmbedder(collection="test_collection", repository=mock_repository)


@pytest.mark.asyncio
async def test_truncate_text_to_max_tokens(embedder):
    """
    Test the _truncate_text_to_max_tokens method truncates properly.
    """
    with patch("zmongo_toolbag.zmongo_embedder.tiktoken.get_encoding") as mock_get_encoding:
        mock_encoding = MagicMock()
        # Suppose our "encoded" text is just a list of token IDs
        mock_encoding.encode.return_value = list(range(9000))  # exceed 8192
        mock_encoding.decode.return_value = "decoded text"
        mock_get_encoding.return_value = mock_encoding

        truncated_text = embedder._truncate_text_to_max_tokens("some long text...")
        assert truncated_text == "decoded text", "Truncated text should match mock decoding."
        assert mock_encoding.encode.call_count == 1, "encode() should be called once."
        assert mock_encoding.decode.call_count == 1, "decode() should be called once."


@pytest.mark.asyncio
async def test_embed_text_returns_cached_embedding(embedder, mock_repository):
    """
    Test that embed_text returns a cached embedding if found in the DB.
    """
    text = "Hello world!"
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    fake_embedding = [0.1, 0.2, 0.3]

    # Mock the repository to return a cached document
    mock_repository.find_document.return_value = {"embedding": fake_embedding}

    with patch("zmongo_toolbag.zmongo_embedder.tiktoken.get_encoding") as mock_get_encoding:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_encoding.decode.return_value = text
        mock_get_encoding.return_value = mock_encoding

        embedding_result = await embedder.embed_text(text)

    mock_repository.find_document.assert_awaited_once_with(
        "_embedding_cache", {"text_hash": text_hash}
    )
    assert embedding_result == fake_embedding, "Should return the cached embedding."


@pytest.mark.asyncio
async def test_embed_text_generates_new_embedding(embedder, mock_repository):
    """
    Test embed_text generates a new embedding if the text is not cached.
    """
    text = "This is a test."
    fake_embedding = [0.4, 0.5, 0.6]

    # No cached doc
    mock_repository.find_document.return_value = None

    with patch.object(embedder.openai_client.embeddings, "create", new=AsyncMock()) as mock_create:
        mock_create.return_value = MagicMock(data=[MagicMock(embedding=fake_embedding)])
        with patch("zmongo_toolbag.zmongo_embedder.tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_encoding.decode.return_value = text
            mock_get_encoding.return_value = mock_encoding

            result = await embedder.embed_text(text)

    assert result == fake_embedding, "Embedding should match what the API returned."
    mock_create.assert_awaited_once()
    mock_repository.insert_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_text_with_invalid_text(embedder):
    """
    Test embed_text raises ValueError if the text is invalid.
    """
    with pytest.raises(ValueError):
        await embedder.embed_text(None)

    with pytest.raises(ValueError):
        await embedder.embed_text(123)  # not a string


@pytest.mark.asyncio
async def test_embed_and_store_valid(embedder, mock_repository):
    """
    Test embed_and_store with valid ObjectId and text.
    """
    doc_id = ObjectId()
    text = "Some text to embed"
    fake_embedding = [0.7, 0.8]

    with patch.object(ZMongoEmbedder, "embed_text", new=AsyncMock(return_value=fake_embedding)) as mock_embed_text:
        await embedder.embed_and_store(doc_id, text, embedding_field="embedding")

        mock_embed_text.assert_awaited_once_with("Some text to embed")

    mock_repository.save_embedding.assert_awaited_once_with(
        "test_collection", doc_id, fake_embedding, "embedding"
    )


@pytest.mark.asyncio
async def test_embed_and_store_invalid_doc_id(embedder):
    """
    Test embed_and_store raises ValueError if document_id is not an ObjectId,
    or if text is empty.
    """
    invalid_doc_id = "not-an-object-id"
    text = "Some text"

    with pytest.raises(ValueError):
        await embedder.embed_and_store(invalid_doc_id, text)

    with pytest.raises(ValueError):
        await embedder.embed_and_store(ObjectId(), "")


@pytest.mark.asyncio
async def test_embed_text_openai_error(embedder, mock_repository):
    """
    Test that embed_text properly raises an exception if OpenAI call fails.
    """
    text = "Test text"
    mock_repository.find_document.return_value = None

    with patch.object(embedder.openai_client.embeddings, "create", new=AsyncMock(side_effect=Exception("OpenAI error"))):
        with pytest.raises(Exception) as exc_info:
            await embedder.embed_text(text)

        assert "OpenAI error" in str(exc_info.value), "Should raise exception from OpenAI API failure."


# Additional tests

@pytest.mark.asyncio
async def test_embed_and_store_exception_logging(embedder, mock_repository):
    """
    Test that embed_and_store logs an error and re-raises the exception.
    """
    from bson.objectid import ObjectId

    doc_id = ObjectId()
    text = "some text"
    fake_exception = Exception("Boom")

    with patch.object(embedder, "embed_text", new=AsyncMock(side_effect=fake_exception)) as mock_embed_text, \
         patch("zmongo_toolbag.zmongo_embedder.logger.error") as mock_logger:
        with pytest.raises(Exception) as exc_info:
            await embedder.embed_and_store(doc_id, text, embedding_field="embedding")

        assert "Boom" in str(exc_info.value)
        mock_logger.assert_called_once()
        logged_message = mock_logger.call_args[0][0]
        assert f"Failed to embed and store text for document {doc_id}" in logged_message
        assert "Boom" in logged_message

    mock_embed_text.assert_awaited_once_with(text)
    mock_repository.save_embedding.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_text_raises_value_error_if_response_data_empty(embedder):
    """
    Test that embed_text raises a ValueError when 'response.data' is empty.
    """

    class FakeEmptyResponse:
        data = []

    with patch.object(embedder.openai_client.embeddings, "create", new=AsyncMock()) as mock_create:
        mock_create.return_value = FakeEmptyResponse()

        with pytest.raises(ValueError) as exc_info:
            await embedder.embed_text("Testing empty data response")

        assert "OpenAI embedding response is empty; expected at least one embedding." in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_text_raises_value_error_if_missing_embedding_field(embedder):
    """
    Test that embed_text raises a ValueError if the first record
    in response.data is missing the 'embedding' attribute.
    """
    class NoEmbedding:
        pass

    class FakeResponse:
        data = [NoEmbedding()]

    with patch.object(embedder.openai_client.embeddings, "create", new=AsyncMock()) as mock_create:
        mock_create.return_value = FakeResponse()

        with pytest.raises(ValueError) as exc_info:
            await embedder.embed_text("Testing missing embedding field")

        assert "OpenAI response is missing embedding data." in str(exc_info.value)