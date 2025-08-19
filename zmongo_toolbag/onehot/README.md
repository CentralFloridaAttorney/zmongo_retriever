# One-Hot Words (Mongo)

Async, SafeResult-based one-hot vocabulary using **ZMongo**.

## Quickstart

```python
import asyncio
from zmongo_toolbag.onehot import MongoOneHotDB

async def run():
    db = MongoOneHotDB(collection="onehot_words")
    await db.init()

    await db.add_word("cat")
    await db.add_word("dog")

    print("size:", (await db.size()).data)
    print("dog index:", (await db.get_index("dog")).data)
    print("index 0 word:", (await db.get_word(0)).data)
    print("all words:", (await db.words()).data)

    print("one-hot('cat'):", (await db.to_one_hot_vector("cat")).data)
    print("bow(['cat','dog','cat']):", (await db.to_bow_vector(['cat','dog','cat'])).data)

asyncio.run(run())
