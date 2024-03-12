# ZMongoRetriever Tutorial: Understanding Key Selection

`ZMongoRetriever` simplifies fetching, processing, and encoding documents from MongoDB, making it invaluable for projects involving large text datasets. A crucial aspect of its functionality is identifying the correct index for a JSON key to specify which part of the document should be processed. This tutorial explains that process in detail.

## Introduction to JSON Keys and Index Selection

When working with nested JSON data stored in MongoDB, each field or value can be accessed using a key or a sequence of keys. However, when data is deeply nested, figuring out the correct key or index for a specific piece of information becomes challenging. `ZMongoRetriever` utilizes a dot-separated string, representing the "path" through the JSON structure to the desired data, as a key sequence.

### Step 1: Retrieve and Examine Your JSON Structure

Before selecting a key, you must understand the structure of your JSON documents. Here's an example of a nested JSON document:

```json
{
  "_id": "12345",
  "report": {
    "summary": "This is a summary of the report.",
    "details": {
      "content": "Detailed content of the report...",
      "author": "John Doe"
    }
  }
}
```

### Step 2: Convert JSON to Metadata

`ZMongoRetriever` can convert your JSON document into a flat metadata dictionary, where each key represents a path to a value in the original JSON structure. The conversion process is handled by the `convert_json_to_metadata` function:

```python
metadata = convert_json_to_metadata(your_json_document)
```

After conversion, the metadata for the above JSON might look like this:

```python
{
  "_id": "12345",
  "report.summary": "This is a summary of the report.",
  "report.details.content": "Detailed content of the report...",
  "report.details.author": "John Doe"
}
```

### Step 3: Identifying the Correct Index

To select a specific field's content as `page_content`, you need to identify its corresponding key in the metadata. For example, if you want to use the report's content, the key would be `"report.details.content"`. However, `ZMongoRetriever` requires you to specify this selection as an index, which refers to the position of the key in a list of all keys sorted alphabetically or as they appear in the document.

To simplify this, use the `get_keys_from_json` function to retrieve all keys and find the index of your desired key:

```python
keys = get_keys_from_json(your_json_document)
index_of_desired_key = keys.index("report.details.content")
```

### Step 4: Using the Index in `ZMongoRetriever`

With the index determined, you can now use it to specify the `page_content_key_index` when invoking `ZMongoRetriever`:

```python
documents = retriever.invoke(object_ids=["12345"], page_content_key_index=index_of_desired_key)
```

This tells `ZMongoRetriever` to use the content found at the specified index as the `page_content` for further processing or encoding.

## Conclusion

Identifying the correct index for a key in nested JSON documents allows you to precisely control which data `ZMongoRetriever` processes. This method provides flexibility and precision, especially in complex datasets. By following the steps outlined above, users can effectively navigate their JSON structures and leverage `ZMongoRetriever` for efficient document processing.
