---

# Using `page_content_key` with `ZMongoRetriever`

When working with nested MongoDB documents, `ZMongoRetriever` allows you to extract specific fields from documents using the `page_content_key` parameter. This key uses **dot notation** to reference deeply nested fields and is passed as an argument when invoking the retriever. This guide explains how to use `page_content_key` to control what gets selected as the main content of a document.

---

## 🔑 What is a `page_content_key`?

A `page_content_key` is a string path that identifies a specific field in a MongoDB document using dot notation. It can reference top-level fields or fields nested within subdocuments and arrays.

For example, in a document like:

```json
{
  "_id": "abc123",
  "title": "The Great Gatsby",
  "author": {
    "firstName": "F. Scott",
    "lastName": "Fitzgerald"
  },
  "published": {
    "year": 1925,
    "publisher": "Charles Scribner's Sons"
  },
  "genres": ["novel", "fiction", "literature"]
}
```

The valid `page_content_key` values might include:

- `"title"`
- `"author.firstName"`
- `"published.publisher"`
- `"genres.0"` (first item in the array)

---

## 🛠️ Using `page_content_key` in Your Code

Here's how you use `page_content_key` when calling the retriever:

```python
page_content_key = "published.publisher"
document = retriever.get_zdocuments(
    object_ids="abc123",
    page_content_key_index=page_content_key
)
print(document.page_content)
```

### ✅ Output:
```
Charles Scribner's Sons
```

---

## 📘 More Examples

### Top-Level Field

**Goal:** Extract the book title.

```python
page_content_key = "title"
```

### Nested Field

**Goal:** Extract the author’s first name.

```python
page_content_key = "author.firstName"
```

### Array Element

**Goal:** Extract the first genre.

```python
page_content_key = "genres.0"
```

---

## 💡 Tips for Choosing a `page_content_key`

1. **Use a MongoDB viewer** like Compass to inspect your document structure.
2. **Use dot notation** for nested keys: `field.subfield`.
3. **Test in MongoDB shell** to make sure your key path resolves to a value.
4. **Handle missing values** gracefully—make sure the field is present or fallback logic is applied.

---

With `page_content_key`, you can precisely control what text is returned for indexing, retrieval, or summarization in pipelines using `ZMongoRetriever`.
