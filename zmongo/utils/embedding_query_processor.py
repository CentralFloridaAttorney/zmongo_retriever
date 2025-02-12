import ast

import openai
import numpy as np
import pandas as pd
import tiktoken
from scipy import spatial
from datetime import datetime

import zconstants
from zmongo_support.utils.data_processing import DataProcessing
from zmongo_support.utils.zmongo_retriever import ZMongoRetriever
from zmongo_support.zmongo_embedder import ZMongoEmbedder


class EmbeddingQueryProcessor:
    def __init__(self, collection_name: str, page_content_key: str):
        """
        Initialize the EmbeddingQueryProcessor with a MongoDB collection name.

        Args:
            collection_name (str): The name of the MongoDB collection containing the text and embeddings.
            page_content_key (str): The key to access the text content in the collection documents.
        """
        self.collection_name = collection_name
        self.page_content_key = page_content_key

        zmongo_retriever = ZMongoRetriever()
        self.df = zmongo_retriever.get_collection_as_dataframe(collection_name=f"{self.collection_name}_embeddings")

        if self.df.empty:
            print(f"Embeddings for {self.collection_name} not found. Generating new embeddings...")
            self._generate_embeddings()
            self.df = zmongo_retriever.get_collection_as_dataframe(collection_name=f"{self.collection_name}_embeddings")

        embedding_list = DataProcessing.get_values_as_list(df=self.df, prefix="embedding_")
        self._process_embeddings(embedding_list)

        self.client = openai.OpenAI()

    def _generate_embeddings(self):
        """Generate embeddings for the collection if they do not already exist."""
        embedder = ZMongoEmbedder(
            collection_to_embed=self.collection_name,
            page_content_key=self.page_content_key
        )
        embedder.embed_collection(page_content_key=self.page_content_key)

    def _process_embeddings(self, embedding_list):
        """Process and reshape embeddings."""
        num_rows = len(self.df)
        embedding_length = len(embedding_list) // num_rows

        if embedding_length * num_rows != len(embedding_list):
            raise ValueError("The total number of embedding values is not evenly divisible by the number of rows.")

        reshaped_embeddings = np.array(embedding_list, dtype=np.float32).reshape(num_rows, embedding_length).tolist()
        self.df['embedding'] = reshaped_embeddings

        if isinstance(self.df['embedding'].iloc[0], str):
            self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def _rank_strings_by_relatedness(self, query: str, top_n: int = 100):
        """Returns a list of text strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding, dtype=np.float32)

        strings_and_relatednesses = []

        for value, row in self.df.iterrows():
            embedding_value = row["embedding"]
            text_value = row["text"]  # Get the text associated with the embedding
            query_embedding_value = np.array(query_embedding, dtype=np.float32)
            row_embedding_value = np.array(embedding_value, dtype=np.float32)

            # Calculate the cosine similarity
            similarity = 1 - spatial.distance.cosine(query_embedding_value, row_embedding_value)

            # Debugging: Print or log the values
            print(f"Value: {value}")
            print(f"Text: {text_value}")
            print(f"Row Embedding: {row_embedding_value}")
            print(f"Query Embedding: {query_embedding_value}")
            print(f"Similarity: {similarity}\n")

            # Append the text and similarity result to the list
            strings_and_relatednesses.append((text_value, similarity))

        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def _num_tokens(self, text: str, model: str = "text-embedding-ada-002") -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def generate_query_message(self, query: str, model: str, token_budget: int) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatedness = self._rank_strings_by_relatedness(query)
        introduction = 'Use the below articles to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        question = f"\n\nQuestion: {query}"
        message = introduction

        for string in strings:
            next_article = f'\n\nRelevant section:\n"""\n{string}\n"""'
            if self._num_tokens(message + next_article + question, model=model) > token_budget:
                break
            else:
                message += next_article
        return message + question

    def chat_with_gpt(self, prompt: str, model: str = "gpt-4o-mini"):
        """Stream a chat completion from GPT and print the response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end='')

        print()  # Ensure the final output ends with a newline


def main():
    processor = EmbeddingQueryProcessor(collection_name=zconstants.ZCASES_COLLECTION,
                                        page_content_key=zconstants.TEST_PAGE_CONTENT_KEY)

    answer = processor.generate_query_message(
        query="Who committed fraud",
        model="text-embedding-ada-002",
        token_budget=4000
    )
    print(answer)


if __name__ == '__main__':
    main()
