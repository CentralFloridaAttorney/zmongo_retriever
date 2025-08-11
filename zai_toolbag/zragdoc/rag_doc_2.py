from models.llama_model import LlamaModel
from zmongo_toolbag.zmongo import ZMongo
from bson import ObjectId
from pydantic import BaseModel, Field

# ---- RAGDoc schema ----
class RAGDoc(BaseModel):
    doc_id: str = Field(..., alias="_id")
    title: str
    content: str
    meta: dict = Field(default_factory=dict)
    embedding: list = Field(default=None, alias="_embedding")
    model_config = {"populate_by_name": True}

def main():
    llama_model = LlamaModel()
    zm = ZMongo()
    collection_name = "rag_encounters"

    prompts = [
        (
            "Tavern Awakening",
            "Write a Dungeons & Dragons encounter using D20 rules. "
            "Include full descriptive text for the dungeon master to read when running the encounter. "
            "This is for new dungeon masters. "
            "The adventurers awake from a drunken slumber in the corner of a tavern."
        ),
        (
            "Forest Ambush",
            "Write a Dungeons & Dragons encounter using D20 rules. "
            "Include full descriptive text for the dungeon master to read aloud. "
            "The adventurers are ambushed by goblins while traveling through a misty forest at dawn."
        )
    ]

    docs = []
    for title, user_input in prompts:
        prompt = llama_model.generate_prompt_from_template(user_input)
        output_text = llama_model.generate_text(
            prompt=prompt,
            max_tokens=3000,
        )
        doc = RAGDoc(
            _id=str(ObjectId()),
            title=title,
            content=output_text,
            meta={"prompt": user_input}
        )
        docs.append(doc)

    # Insert all docs to Mongo using ZMongo, by alias
    import asyncio

    async def insert_all():
        for doc in docs:
            res = await zm.insert_document(collection_name, doc.model_dump(by_alias=True))
            assert res.success
            print(f"Inserted: {doc.title} ({doc.doc_id})")

    asyncio.run(insert_all())

    print("\nAll encounters generated and saved!")

if __name__ == "__main__":
    main()
