# main.py

import asyncio
from bson import ObjectId

from zmongo_retriever.examples.openai_model import OpenAIModel


async def main():
    model = OpenAIModel()

    # 👤 Example instruction
    instruction = "Explain the concept of async programming in Python."
    response = await model.generate_instruction(instruction)
    print("\n🔹 Instruction Response:\n", response)

    # 📄 Example summary
    long_text = (
        "Asynchronous programming in Python allows developers to write code that can pause "
        "while waiting for an operation to complete (like a network request) and continue "
        "executing other code during that time. It enables better scalability and performance "
        "in I/O-bound programs. Libraries like asyncio make this possible."
    )
    summary = await model.generate_summary(long_text)
    print("\n🔹 Summary:\n", summary)

    # ❓ Question answering
    context = "Python supports async programming using the asyncio library and the 'async' and 'await' keywords."
    question = "How does Python support asynchronous programming?"
    answer = await model.generate_question_answer(context, question)
    print("\n🔹 Q&A:\n", answer)

    # 🧬 ZElement explanation
    zelement_doc = {
        "name": "Case Precedent Extractor",
        "note": "Designed to retrieve and summarize legal precedents from MongoDB based on user queries.",
        "creator": "LegalAI-Labs"
    }
    explanation = await model.generate_zelement_explanation(zelement_doc)
    print("\n🔹 ZElement Explanation:\n", explanation)

    # 🧾 Save summary to MongoDB (optional demo)
    # Replace with a real _id and ensure the collection exists
    fake_id = ObjectId()  # Replace with real ID if testing on real DB
    saved = await model.save_openai_result(
        collection_name="documents",
        record_id=fake_id,
        field_name="ai_summary",
        generated_text=summary,
        extra_fields={"ai_summary_source": "OpenAI gpt-3.5-turbo-instruct"}
    )
    print("\n✅ Saved to MongoDB:", saved)


if __name__ == "__main__":
    asyncio.run(main())
