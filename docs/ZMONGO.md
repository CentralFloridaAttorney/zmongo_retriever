# ZMongo: Async MongoDB Toolkit for Python

`ZMongo` is a high-level, async Python MongoDB interface with built-in data normalization, Pydantic support, bulk operations, and auto-handling for underscore-key aliasing.
**Every method returns a [SafeResult](#saferesult) with success/error/data.**

---

## **Quickstart**

```python
from zmongo_toolbag.zmongo import ZMongo
zm = ZMongo()
```

---

## **Table of Contents**

* [Inserting Documents](#inserting-documents)
* [Finding Documents](#finding-documents)
* [Updating Documents](#updating-documents)
* [Deleting Documents](#deleting-documents)
* [Bulk Operations](#bulk-operations)
* [Aggregation](#aggregation)
* [Counting Documents](#counting-documents)
* [Listing Collections](#listing-collections)
* [Sorting](#sorting)
* [Key Aliasing and Restoration](#key-aliasing-and-restoration)
* [Handling Empty/Edge Cases](#handling-emptyedge-cases)
* [SafeResult](#saferesult)
* [Advanced: Bulk Write and Aggregation Limits](#advanced-bulk-write-and-aggregation-limits)

---

## **Inserting Documents**

### Insert a Single Document (dict or Pydantic model)

```python
import asyncio
from zmongo_toolbag.zmongo import ZMongo

async def insert_one():
    zm = ZMongo()
    coll = "pets"
    doc = {"name": "Whiskers", "age": 3, "_secret": "purr"}
    res = await zm.insert_document(coll, doc)
    assert res.success
    print(res.data.inserted_id)  # The MongoDB ObjectId

# asyncio.run(insert_one())
```

### Insert a Pydantic Model

```python
from pydantic import BaseModel, Field

class Pet(BaseModel):
    name: str
    age: int
    secret: str = Field(..., alias="_secret")

pet = Pet(name="Rex", age=5, _secret="bowwow")
res = await zm.insert_document(coll, pet)
```

### Insert Many Documents

```python
pets = [Pet(name=f"pet{i}", age=i, _secret=f"s{i}") for i in range(5)]
ins = await zm.insert_documents(coll, pets)
assert ins.success
print(ins.data.inserted_ids)
```

---

## **Finding Documents**

### Find One

```python
res = await zm.find_document(coll, {"name": "Whiskers"})
if res.success:
    doc = res.data
    print(doc["name"])
```

### Find Many

```python
res = await zm.find_documents(coll, {"age": {"$gte": 3}}, limit=10)
for doc in res.data:
    print(doc)
```

---

## **Updating Documents**

### Update One Document

```python
upd = await zm.update_document(coll, {"name": "Buddy"}, {"age": 2, "_secret": "yap"})
assert upd.success
```

### Update Many (with upsert)

```python
result = await zm.update_documents(coll, {"role": "user"}, {"role": "admin"})
assert result.success

# Upsert: update or insert if not found
upsert_result = await zm.update_documents(
    coll, {"name": "carol"}, {"role": "user", "name": "carol"}, upsert=True
)
assert upsert_result.success
```

---

## **Deleting Documents**

### Delete One

```python
del_one = await zm.delete_document(coll, {"name": "Whiskers"})
assert del_one.success
```

### Delete Many

```python
del_all = await zm.delete_documents(coll)
assert del_all.success
```

---

## **Bulk Operations**

```python
from pymongo.operations import InsertOne, DeleteMany
ops = [InsertOne({"foo": i}) for i in range(5)]
bulk_res = await zm.bulk_write(coll, ops)
assert bulk_res.success
```

---

## **Aggregation**

```python
pipe = [{"$group": {"_id": "$cat", "count": {"$sum": 1}}}]
agg = await zm.aggregate(coll, pipe)
for row in agg.data:
    print(row["_id"], row["count"])
```

* Limit results from an aggregation:

  ```python
  agg = await zm.aggregate(coll, pipe, limit=5)
  assert len(agg.data) <= 5
  ```

---

## **Counting Documents**

```python
cnt = await zm.count_documents(coll, {"cat": "A"})
print(cnt.data["count"])
```

---

## **Listing Collections**

```python
result = await zm.list_collections()
print(result.data)  # List of collection names
```

---

## **Sorting Results**

```python
# Find all docs, ascending by value
result = await zm.find_documents(coll, {}, sort=[("value", 1)])
asc_values = [doc["value"] for doc in result.data]
# Descending:
result = await zm.find_documents(coll, {}, sort=[("value", -1)])
desc_values = [doc["value"] for doc in result.data]
```

---

## **Key Aliasing and Restoration**

* Fields starting with `_` are aliased for Mongo compatibility.
* After retrieval, keys are restored to their original names.

**Example:**

```python
doc = {"name": "Tiger", "age": 7, "_secret": "stripe"}
await zm.insert_document(coll, doc)
res = await zm.find_document(coll, {"name": "Tiger"})
assert res.data["_secret"] == "stripe"
```

---

## **Handling Empty/Edge Cases**

* **Insert empty list:** returns success, with empty `inserted_ids`.
* **Aggregate with limit higher than doc count:** returns all docs, not more.
* **Non-string ObjectIds** are handled gracefully (see SafeResult).

---

## **SafeResult**

Every operation returns a `SafeResult`:

* `.success` (bool): Was the operation successful?
* `.data`: The data payload (document, list, or primitive)
* `.error`: The error message, or `None`
* `.original()`: Restore original document key names and ObjectIds if needed
* `.model_dump()`: JSON-serializable representation
* `.to_json()`: Dump as JSON string

**Example:**

```python
res = await zm.find_document(coll, {"name": "Whiskers"})
if res.success:
    print(res.data)        # Use dict keys as normal
    print(res.original())  # Restore _id as ObjectId, handle "__keymap"
```

---

## **Advanced: Bulk Write and Aggregation Limits**

* **Bulk write:** Accepts any valid PyMongo bulk ops
* **Aggregation:** `limit` parameter enforces early cutoff (even if aggregation would yield more results)

---

## **Tested Patterns (from test suite)**

* Insert/find by dict or Pydantic model
* Insert/find many documents
* Update with `$set` or whole document
* Upsert (insert if not found)
* Delete one/many docs
* Bulk write and delete
* Aggregation with and without limits
* Sorting
* Key alias collision handling (see `_sanitize_dict`)
* List all collections
* Return empty on insert empty list

---

## **Troubleshooting**

* Always check `.success` before accessing `.data`
* Use `.original()` for legacy or strict key-name needs
* Clean up test collections after use
* For custom models, ensure correct aliasing in Pydantic

---

**ZMongo** streamlines async MongoDB usage in Python, with reliability and schema-safe results.
For more advanced recipes, consult the tests or source code!

