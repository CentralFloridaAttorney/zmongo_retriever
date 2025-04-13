import asyncio
from bson import ObjectId
from zmongo_toolbag.models.llama_model import LlamaModel  # adjust path as needed
from zmongo_toolbag.zmongo import ZMongo


async def test_llama_model_save():
    # Setup
    zmongo = ZMongo()
    collection_name = "llama_model_test_collection"

    # Insert a dummy doc to update
    dummy_doc = {"placeholder": True}
    inserted = await zmongo.insert_document(collection_name, dummy_doc)
    doc_id = inserted.inserted_id
    print(f"[+] Inserted dummy doc with _id={doc_id}")

    # Create LlamaModel instance
    model = LlamaModel(zmongo=zmongo)

    # Simulate generated text
    generated_text = "This is a test response from LlamaModel."
    field_name = "ai_response"
    extra_fields = {"test_meta": "real_data_test"}

    # Perform the save
    result = await model.save(
        collection_name=collection_name,
        record_id=doc_id,
        field_name=field_name,
        generated_text=generated_text,
        extra_fields=extra_fields
    )

    assert result, "Save failed!"

    # Verify in DB
    updated_doc = await zmongo.find_document(collection_name, {"_id": doc_id})
    print("[+] Updated document:")
    print(updated_doc)

    assert updated_doc.get(field_name) == generated_text, "Field was not updated correctly."
    assert updated_doc.get("test_meta") == "real_data_test", "Extra field was not saved."

    # Cleanup
    await zmongo.delete_document(collection_name, {"_id": doc_id})
    print("[+] Cleanup successful.")


if __name__ == "__main__":
    asyncio.run(test_llama_model_save())
