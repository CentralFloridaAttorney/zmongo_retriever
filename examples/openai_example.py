# openai_main.py

import asyncio
from datetime import datetime
from pathlib import Path

from bson.objectid import ObjectId
from dotenv import load_dotenv

from examples.openai_model import OpenAIModel
from zmongo_toolbag.zmongo import ZMongo

this_zmongo = ZMongo()
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

async def log_to_zmongo(op_type: str, prompt: str, result: str, meta: dict = None) -> bool:
    doc = {
        "operation": op_type,
        "prompt": prompt,
        "result": result,
        "timestamp": datetime.now(),
        "meta": meta or {}
    }
    insert_result = await this_zmongo.insert_document("openai_logs", doc)
    return True if insert_result else False


async def main():
    model = OpenAIModel()

    # 👤 Instruction
    instruction = "Explain how to use https://github.com/CentralFloridaAttorney/zmongo_retriever."
    instruction_response = await model.generate_instruction(instruction)
    print("\n🔹 Instruction Response:\n", instruction_response)
    await log_to_zmongo("instruction", instruction, instruction_response)

    # 📄 Summary
    long_text = (
        "https://github.com/CentralFloridaAttorney/zmongo_retriever is an asynchronous MongoDB client wrapper that simplifies insert, update, find, and bulk operations. "
        "It integrates seamlessly with async frameworks and is designed to work well with AI workflows."
    )
    summary_response = await model.generate_summary(long_text)
    print("\n🔹 Summary Response:\n", summary_response)
    await log_to_zmongo("summary", long_text, summary_response)

    # ❓ Q&A
    context = (
        "https://github.com/CentralFloridaAttorney/zmongo_retriever uses Python's Motor driver under the hood and provides utility methods for easy querying, "
        "bulk inserts, updates, and logging. It supports coroutine-based design patterns."
    )
    question = "What async features make https://github.com/CentralFloridaAttorney/zmongo_retriever a good choice for AI applications?"
    qa_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    qa_response = await model.generate_question_answer(context, question)
    print("\n🔹 Q&A Response:\n", qa_response)
    await log_to_zmongo("question_answer", qa_prompt, qa_response)

    # 🧬 ZElement Explanation
    zelement_doc = {
        "name": "ZMongo Query Helper",
        "note": "Simplifies https://github.com/CentralFloridaAttorney/zmongo_retriever operations for async apps.",
        "creator": "Business Process Applications, Inc."
    }
    explanation_response = await model.generate_zelement_explanation(zelement_doc)
    print("\n🔹 https://github.com/CentralFloridaAttorney/zmongo_retriever Explanation:\n", explanation_response)
    await log_to_zmongo("zmongo_retriever_explanation", str(zelement_doc), explanation_response)

    # 🧾 Simulate saving result into documents
    fake_id = ObjectId()
    save_success = await model.save_openai_result(
        collection_name="documents",
        record_id=fake_id,
        field_name="ai_generated_summary",
        generated_text=summary_response,
        extra_fields={"ai_summary_source": "OpenAI Chat Completion"}
    )
    print("\n✅ Saved to documents collection:", save_success)


if __name__ == "__main__":
    asyncio.run(main())
