import asyncio
import logging
from typing import Optional, List

import numpy as np
import openai
import tiktoken
from scipy import spatial

from zmongo.utils.data_processing import DataProcessing
from zmongo import zconstants
from zmongo.zmongo_repository import ZMongoRepository
from zmongo.zmongo_retriever import ZMongoEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingQueryProcessor:
    def __init__(self, collection_name: str, page_content_keys: List[str]):
        """
        Initialize the EmbeddingQueryProcessor with a MongoDB collection name and multiple page content keys.

        Args:
            collection_name (str): The name of the MongoDB collection containing the text and embeddings.
            page_content_keys (List[str]): List of dot-separated paths to the content fields.
        """
        self.collection_name = collection_name
        self.page_content_keys = page_content_keys
        self.repository = ZMongoRepository()
        self.embeddings = {}  # Dictionary to store embeddings per content key
        self.texts = {}       # Dictionary to store texts per content key

    async def initialize(self):
        """Asynchronously initialize embeddings."""
        await self._initialize_embeddings()

    async def _initialize_embeddings(self):
        """Fetch existing embeddings or generate new ones if not available."""
        for content_key in self.page_content_keys:
            embedding_field = f"embeddings.{content_key.replace('.', '_')}"
            projection = {"_id": 1, content_key: 1, embedding_field: 1}
            documents = await self.repository.find_documents(
                collection=self.collection_name,
                query={},
                projection=projection
            )
            if not documents:
                logger.info(f"No documents found in collection '{self.collection_name}'.")
                continue

            # Collect IDs of documents missing embeddings
            missing_embedding_ids = []
            for doc in documents:
                doc_json = DataProcessing.convert_object_to_json(doc)
                embedding_value = DataProcessing.get_value(doc_json, content_key)
                if content_key not in doc or embedding_value is None:
                    missing_embedding_ids.append(str(doc._id))

            # Generate embeddings for missing documents
            if missing_embedding_ids:
                logger.info(f"Generating embeddings for {len(missing_embedding_ids)} documents missing embeddings for content key '{content_key}'.")
                embedder = ZMongoEmbedder(
                    page_content_keys=[content_key],
                    collection_name=self.collection_name,
                    embedding_model=zconstants.EMBEDDING_MODEL,
                    max_tokens_per_chunk=128,  # Adjust as needed
                    overlap_prior_chunks=50,    # Adjust as needed
                    encoding_name=zconstants.EMBEDDING_ENCODING,
                    openai_api_key=zconstants.OPENAI_API_KEY,
                )
                await embedder.process_documents(missing_embedding_ids)

                # Reload documents to include newly generated embeddings
                documents = await self.repository.find_documents(
                    collection=self.collection_name,
                    query={},
                    projection=projection
                )

            # Initialize lists for this content key
            self.embeddings[content_key] = []
            self.texts[content_key] = []

            for doc in documents:
                doc_json = DataProcessing.convert_object_to_json(doc)
                embedding_value = DataProcessing.get_value(doc_json, embedding_field)
                texts = DataProcessing.get_value(doc_json, content_key)
                if texts is not None and embedding_value is not None:
                    self.embeddings[content_key].append(embedding_value)
                    self.texts[content_key].append(texts)
                else:
                    logger.warning(f"Embedding for document ID {doc['_id']} and content key '{content_key}' not found even after generation.")

    async def _rank_strings_by_relatedness(self, query: str, top_n: int = 100, content_key: Optional[str] = None):
        """
        Return a list of text strings and relatednesses, sorted from most related to least, for a specific content key.

        Args:
            query (str): The query string.
            top_n (int): Number of top related strings to return.
            content_key (Optional[str]): The specific content key to rank against. If None, ranks across all content keys.

        Returns:
            Tuple[List[str], List[float]]: Tuple of lists containing the top related strings and their similarity scores.
        """
        # Generate query embedding
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding = self.get_embedding_from_response(response)

        strings_and_relatednesses = []

        keys_to_process = [content_key] if content_key else self.page_content_keys

        for key in keys_to_process:
            embeddings = self.embeddings.get(key, [])
            texts = self.texts.get(key, [])
            if not embeddings or not texts:
                logger.warning(f"No embeddings or texts found for content key '{key}'.")
                continue
            for embedding, text in zip(embeddings, texts):
                similarity = 1 - spatial.distance.cosine(query_embedding, np.array(embedding, dtype=np.float32))
                strings_and_relatednesses.append((text, similarity))

        # Sort and select top_n
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        if strings_and_relatednesses:
            top_strings, top_relatednesses = zip(*strings_and_relatednesses[:top_n])
            return list(top_strings), list(top_relatednesses)
        else:
            return [], []

    def _num_tokens(self, text: str, model: str = "text-embedding-ada-002") -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    async def generate_query_message(self, query: str, model: str, token_budget: int) -> str:
        """
        Return a message for GPT, with relevant source texts pulled from the collection.

        Args:
            query (str): The question to be answered.
            model (str): The OpenAI model to use.
            token_budget (int): The maximum number of tokens allowed in the prompt.

        Returns:
            str: The constructed message containing relevant sections and the question.
        """
        introduction = 'Use the below articles to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        question = f"\n\nQuestion: {query}"
        message = introduction

        # Iterate over each content key to gather relevant sections
        for content_key in self.page_content_keys:
            strings, _ = await self._rank_strings_by_relatedness(query, top_n=100, content_key=content_key)
            if not strings:
                logger.warning(f"No relevant strings found for content key '{content_key}'.")
                continue
            for string in strings:
                next_article = f'\n\nRelevant section from "{content_key}":\n"""\n{string}\n"""'
                if self._num_tokens(message + next_article + question, model=model) > token_budget:
                    break
                else:
                    message += next_article

        return message + question

    @staticmethod
    def get_embedding_from_response(response) -> List[float]:
        """
        Extracts the embedding vector from the OpenAI API response.
        """
        # Access the data using attribute access
        embedding = response.data[0].embedding
        return embedding

    async def chat_with_gpt(self, prompt: str, model: str = "gpt-4"):
        """
        Send a chat completion request to GPT and print the response.

        Args:
            prompt (str): The prompt to send to GPT.
            model (str): The OpenAI model to use.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )

        print("GPT Response:\n")
        message = response.choices[0].message.content
        print(message)


async def main():
    # List of content keys (dot-separated paths)
    page_content_keys = [
        'meaning_upright',
        'meaning_reversed',
    ]

    # Initialize EmbeddingQueryProcessor
    processor = EmbeddingQueryProcessor(
        collection_name="tarot_cards",
        page_content_keys=page_content_keys
    )

    # Initialize embeddings
    await processor.initialize()

    # Generate query message
    answer_message = await processor.generate_query_message(
        query="Which 5 cards are the best to have in a tarot spread when the person wants family harmony",
        model="text-embedding-ada-002",
        token_budget=4000
    )
    print(answer_message)

    # Optionally, chat with GPT using the generated message
    await processor.chat_with_gpt(prompt=answer_message, model="gpt-4")


if __name__ == '__main__':
    asyncio.run(main())
