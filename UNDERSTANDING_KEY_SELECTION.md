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

`ZMongoRetriever` adds creates a `Document` with `page_content` from a specified field from a record in a mongo collection.  When using `ZMongoRetriever.invoke(object_ids, page_content_key, existing_metadata=None)` you must specify the `page_content_key`, which is a path-like key for the field that is to be used when creating the `page_content` in the `Document`.  Otherwise, you will get my default key, which will probably not work with your data.

`ZMongoRetriever` converts the mongo record into a flattened metadata dictionary, where each key represents a path to a value in the original JSON structure. The conversion process is handled by the `convert_json_to_metadata` function:

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

To select a specific field to be used as the`page_content`, you need to identify its corresponding key in the metadata. For example, if you want to use the report's content, the key would be `"report.details.content"`.  When using `ZMongoRetriever`, you must specify the key for the value to be used when creating Documents.

To simplify the process of extracting information from mongo databases, `json_keys` allows you to get any value using a string like path based  by converting the record into metadata:

```python
# For testing, check your mongo database to get the _id for a record using `system_manager.py` or MongoDBCompass
document = mongo_collection.find_one({'_id': ObjectId('65f1b6beae7cd4d4d1d3ae8d')})
document_metadata = convert_json_to_metadata(document)
# If you retrieved an object from mongodb then it will have an ObjectId('_id')
this_id = document_metadata.get('_id')
self.assertIsInstance(ObjectId(this_id), ObjectId)
# An example of getting the value from the json_key path as shown above:
report_details_content = document_metadata.get('report.details.content')
```
### Step 4: Using the Index in `ZMongoRetriever`

With the index determined, you can now use it to specify the `page_content_key_index` when invoking `ZMongoRetriever`:

```python
documents = retriever.invoke(object_ids=['65f1b6beae7cd4d4d1d3ae8d', '25f1b6beae7cd4d4d1d3ae8s' ], page_content_key='report.details.content')
```

This tells `ZMongoRetriever` to use the content found at the specified index as the `page_content` for further processing or encoding.

## Conclusion

Identifying the correct index for a key in nested JSON documents allows you to precisely control which data `ZMongoRetriever` processes. This method provides flexibility and precision, especially in complex datasets. By following the steps outlined above, users can effectively navigate their JSON structures and leverage `ZMongoRetriever` for efficient document processing.
