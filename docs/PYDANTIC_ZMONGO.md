# Guide: Using Custom Pydantic Models with ZMongo and Safe Underscore Key Aliasing

When working with MongoDB and Python, **field names that start with an underscore (`_`) can create issues**—MongoDB may reserve or mishandle them.
**ZMongo** auto-aliases such keys to ensure safety, and can restore them on retrieval.
**Pydantic** models work seamlessly with this by using field aliases.

This guide explains:

* Why aliasing is needed
* How to define and use Pydantic models with field aliases
* How ZMongo serializes, stores, and restores your data
* Real usage patterns and tips

---

## Why Use Aliasing?

* MongoDB does not recommend user-defined fields starting with `_`.
* ZMongo converts fields like `"_secret"` to a safe, unique key (e.g., `"usecret"`), storing a keymap for later restoration.
* With Pydantic, you can **use aliases to map your Python fields to MongoDB-safe names**—and ZMongo will preserve and restore them as expected.

---

## 1. **Defining a Pydantic Model With Aliases**

Use Pydantic’s `Field(..., alias=...)` to set up aliasing.

```python
from pydantic import BaseModel, Field

class Pet(BaseModel):
    name: str
    age: int
    # This field is "_secret" in Mongo, but "secret" in Python code
    secret: str = Field(..., alias="_secret")
```

Now you can use both the alias and the field name in your code:

```python
pet = Pet(name="Milo", age=5, _secret="hiddenfish")
print(pet.dict(by_alias=True))
# {'name': 'Milo', 'age': 5, '_secret': 'hiddenfish'}
```

---

## 2. **Inserting a Model Into Mongo With ZMongo**

When you pass a Pydantic model to `insert_document`, always use `by_alias=True` if you call `.dict()` directly:

```python
from zmongo_toolbag.zmongo import ZMongo
zm = ZMongo()
coll = "pets"
pet = Pet(name="Nina", age=3, _secret="hushhush")

# Option 1: Let ZMongo handle it
await zm.insert_document(coll, pet)

# Option 2: Explicit .dict(by_alias=True)
await zm.insert_document(coll, pet.dict(by_alias=True))
```

ZMongo will further sanitize any remaining unsafe keys and store a keymap for perfect restoration.

---

## 3. **Retrieving and Restoring Models**

When you retrieve, ZMongo will restore your keys to their original form:

```python
res = await zm.find_document(coll, {"name": "Nina"})
if res.success:
    doc = res.data
    print(doc["_secret"])  # This works!
    # To turn it back into a Pydantic model:
    pet = Pet.parse_obj(doc)
```

If you want original BSON/ObjectId types or key names:

```python
doc = res.original()  # Restores _id as ObjectId, and _secret as "_secret"
pet = Pet.parse_obj(doc)
```

---

## 4. **Bulk Operations With Aliased Models**

```python
pets = [
    Pet(name=f"Pet{i}", age=i, _secret=f"shh{i}") for i in range(3)
]
await zm.insert_documents(coll, [p.dict(by_alias=True) for p in pets])
```

---

## 5. **Practical Patterns and Tips**

* **Always use `by_alias=True` when converting to dict for Mongo.**
* **For legacy code:** If you ever see "usecret" or similar keys, use `.original()` on the SafeResult to restore original names.
* **Use Pydantic validation:** You can validate restored docs directly:

  ```python
  pet = Pet.parse_obj(res.original())
  ```
* **Avoid field names with double underscores or names that would endlessly collide.** ZMongo is robust, but clear field names are better.
* **Custom models with nested structures** are supported; aliases propagate.

---

## 6. **Example: Full Flow**

```python
from zmongo_toolbag.zmongo import ZMongo
from pydantic import BaseModel, Field
import asyncio

class User(BaseModel):
    name: str
    password: str = Field(..., alias="_password")

async def main():
    zm = ZMongo()
    coll = "users"
    user = User(name="alice", _password="hunter2")

    # Insert
    await zm.insert_document(coll, user)

    # Retrieve
    res = await zm.find_document(coll, {"name": "alice"})
    doc = res.original()   # Restores _password!
    user2 = User.parse_obj(doc)
    print(user2)
    assert user2.password == "hunter2"

asyncio.run(main())
```

---

## 7. **Troubleshooting**

* If a restored doc raises a validation error, check your aliases match your Mongo schema.
* If you insert a dict with both "\_secret" and "usecret", ZMongo will avoid collision using multiple "u"s (see tests).
* Always check `.success` on results before accessing `.data` or `.original()`.

---

## 8. **Testing Aliasing**

Use tests to ensure round-trip integrity:

```python
def test_pydantic_alias_roundtrip():
    zm = ZMongo()
    class Model(BaseModel):
        field: str = Field(..., alias="_field")
    obj = Model(_field="val")
    # Insert and retrieve
    res = asyncio.run(zm.insert_document("t", obj))
    doc = asyncio.run(zm.find_document("t", {"field": "val"})).original()
    assert doc["_field"] == "val"
    assert Model.parse_obj(doc).field == "val"
```

---

## **Summary Table**

| Task                | Pydantic Model                                                               | MongoDB Storage | Retrieval       |
| ------------------- | ---------------------------------------------------------------------------- | --------------- | --------------- |
| Field with \_ alias | `secret: Field(..., alias=\"_secret\")`                                      | Safely stored   | Safely restored |
| Insert model        | `await zm.insert_document(coll, model)`                                      | .               | .               |
| Retrieve & restore  | `res = await zm.find_document(...); model = Model.parse_obj(res.original())` | .               | .               |

---

## **Further Reading**

* [Pydantic Aliases](https://docs.pydantic.dev/usage/models/#field-aliases)
* [ZMongo SafeResult & Keymap Handling](#)
* [MongoDB Restrictions on Field Names](https://www.mongodb.com/docs/manual/reference/limits/#Restrictions-on-Field-Names)

