# `EmbeddingQueryProcessor` Script Documentation

This document explains the purpose and functionality of the `EmbeddingQueryProcessor` script. It outlines how the script works, its key components, and its overall workflow.

---

## **Overview**

The script is designed to process embeddings for a MongoDB collection, retrieve relevant sections of text based on a query, and generate a prompt for interaction with OpenAI's GPT models. It handles the following tasks:

1. **Embedding Management:**
   - Fetches existing embeddings from the MongoDB collection.
   - Generates missing embeddings using OpenAI's embedding API.
2. **Query Processing:**
   - Matches a user-provided query against stored embeddings to find the most relevant content.
   - Constructs a message for GPT with the relevant sections.
3. **GPT Interaction:**
   - Sends a prompt to OpenAI's GPT model and prints the response.

---

## **Key Components**

### 1. **Imports and Setup**

The script imports required libraries and modules:
- **`asyncio`**: To handle asynchronous operations.
- **`numpy` and `scipy`**: For numerical operations and similarity calculations.
- **`tiktoken`**: To count tokens in text.
- **`openai`**: For interacting with OpenAI's APIs.
- **Custom modules**:
  - `zmongo.zmongo_embedder`: Handles embedding generation and saving.
  - `zmongo.zmongo_repository`: Interface for MongoDB operations.
  - `zconstants`: Holds constants like API keys and model configurations.

Logging is configured to display informational messages.

---

### 2. **`EmbeddingQueryProcessor` Class**

#### **Initialization**
The constructor initializes the processor with:
- The MongoDB collection name (`collection_name`).
- A list of content keys (`page_content_keys`) that specify the fields in the database containing text content.

It also sets up:
- `ZMongoRepository` for database operations.
- Two dictionaries:
  - `self.embeddings`: Stores embeddings for each content key.
  - `self.texts`: Stores corresponding texts for each content key.

#### **Methods**

1. **`initialize`**:
   - Entry point for asynchronous initialization.
   - Calls `_initialize_embeddings` to fetch or generate embeddings.

2. **`_initialize_embeddings`**:
   - Fetches documents from the MongoDB collection.
   - Checks if embeddings exist for the specified content keys.
   - If embeddings are missing:
     - Uses `ZMongoEmbedder` to generate embeddings.
     - Reloads the documents to include the newly generated embeddings.
   - Stores embeddings and their corresponding texts in `self.embeddings` and `self.texts`.

3. **`_rank_strings_by_relatedness`**:
   - Generates an embedding for a user query using OpenAI's API.
   - Compares the query embedding with stored embeddings using cosine similarity.
   - Ranks texts by their similarity to the query.
   - Returns the top matching texts and their similarity scores.

4. **`_num_tokens`**:
   - Counts the number of tokens in a given text using `tiktoken`.

5. **`generate_query_message`**:
   - Constructs a prompt for GPT models.
   - Includes the most relevant sections of text based on the query, within a specified token limit (`token_budget`).

6. **`chat_with_gpt`**:
   - Sends a prompt to the GPT model and retrieves the response.
   - Prints the GPT-generated answer.

7. **`get_embedding_from_response` (Static Method)**:
   - Extracts the embedding vector from an OpenAI API response.

---

### 3. **`main` Function**

The `main` function orchestrates the following tasks:
1. Defines the content keys to process (e.g., `meaning_upright`, `meaning_reversed`).
2. Initializes the `EmbeddingQueryProcessor`.
3. Generates a query message with the most relevant sections.
4. Interacts with GPT using the generated prompt and prints the response.

---

## **Workflow**

1. **Initialization**:
   - The `EmbeddingQueryProcessor` fetches documents from the MongoDB collection.
   - For each content key:
     - It checks for missing embeddings.
     - Generates missing embeddings and reloads the documents.
     - Stores the embeddings and corresponding texts.

2. **Query Matching**:
   - The user provides a query.
   - The script generates an embedding for the query and calculates its similarity to stored embeddings.
   - The most relevant sections of text are identified.

3. **Prompt Generation**:
   - The script constructs a GPT-compatible prompt that includes relevant sections of text and the user's query.

4. **GPT Interaction**:
   - The prompt is sent to GPT for processing.
   - The GPT-generated response is printed.

---

## **Usage**

### Running the Script
Execute the script using:
```bash
python embedding_query_processor.py
```

### Sample Output
The script retrieves relevant sections, generates a prompt, and interacts with GPT. A typical output might look like:
```
Use the below articles to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."

Relevant section from "meaning_upright":
"""
The upright meaning of the card symbolizes positivity and strength.
"""

Relevant section from "meaning_reversed":
"""
The reversed meaning of the card indicates challenges and obstacles.
"""

Question: Which of the cards is the best card?
GPT Response:
The best card depends on your specific context. Generally, "The Sun" card is considered highly positive.
```

---

## **Key Considerations**

1. **Asynchronous Operations**:
   - Ensure the event loop is running correctly for all `async` methods.
   - Use `asyncio.run` for top-level execution.

2. **MongoDB Connection**:
   - Verify that `ZMongoRepository` connects to the correct database and collection.

3. **OpenAI API**:
   - Ensure `zconstants.OPENAI_API_KEY` is valid and the required models (`text-embedding-ada-002`, `gpt-4`) are available.

4. **Performance**:
   - Embedding generation can be resource-intensive. Monitor and optimize the number of API calls.

5. **Token Limits**:
   - Adjust `token_budget` in `generate_query_message` to fit the model's maximum context length.

---

## **Future Improvements**

- **Batch Processing**:
  - Process embeddings in smaller batches for scalability.
  
- **Error Handling**:
  - Add robust error handling for API calls and database operations.

- **Caching**:
  - Cache embeddings locally to reduce redundant API calls.

---

This documentation provides a comprehensive understanding of the `EmbeddingQueryProcessor` script, making it easier to use, modify, and extend.