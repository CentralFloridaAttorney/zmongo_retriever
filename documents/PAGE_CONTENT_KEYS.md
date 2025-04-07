---

# 🧩 Using `page_content_key` with `ZMongoRetriever`

`ZMongoRetriever` makes it easy to pull structured content from MongoDB—especially when you're dealing with **deeply nested documents**.

The magic lies in one parameter: `page_content_key`. This guide shows you how to use it effectively to pinpoint and extract the **exact field** you want to treat as a document's main content.

---

## 🔑 What Is `page_content_key`?

The `page_content_key` is a **dot-notated string** that tells `ZMongoRetriever` where to look inside your document. Whether the field is top-level or deeply nested inside objects or arrays, you can point to it precisely.

### Example Document

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

### Examples of `page_content_key` Paths

| Field You Want           | `page_content_key` Value        |
|--------------------------|----------------------------------|
| Title                    | `"title"`                       |
| Author’s First Name      | `"author.firstName"`            |
| Publisher Name           | `"published.publisher"`         |
| First Genre in Array     | `"genres.0"`                    |

---

## ⚙️ Using `page_content_key` in Code

Here’s how to pass it to `get_zdocuments`:

```python
page_content_key = "published.publisher"

document = await retriever.get_zdocuments(
    object_ids="abc123",
    page_content_key_index=page_content_key
)

print(document[0].page_content)
```

### ✅ Output:
```
Charles Scribner's Sons
```

---

## 📚 Use Cases

### 🔹 Get a Top-Level Field
```python
page_content_key = "title"  # Outputs: "The Great Gatsby"
```

### 🔹 Fetch a Nested Field
```python
page_content_key = "author.firstName"  # Outputs: "F. Scott"
```

### 🔹 Access an Array Item
```python
page_content_key = "genres.0"  # Outputs: "novel"
```

---

## 💡 Pro Tips

- 🧭 **Explore with MongoDB Compass**  
  Visualize your documents to easily identify valid key paths.

- 📌 **Dot Notation is Key**  
  Use `"outer.inner"` for nested fields, and `"array.0"` for indexed array elements.

- 🧪 **Test Your Keys**  
  Try them in the Mongo shell or Compass to verify that they return expected results.

- 🛑 **Expect the Unexpected**  
  Always check for missing fields or null values in production code.

---

## 🎯 Why Use `page_content_key`?

Because precision matters. Whether you’re indexing for search, embedding for AI workflows, or generating summaries, you want to feed your pipeline the **right** slice of each document.

With `page_content_key`, you're in control.

---
