import asyncio
import os
from dotenv import load_dotenv

# 🔽 Import from installed zmongo-repository package
from zai_model import ZAIModel

load_dotenv()

async def demo_zai_task():
    model = ZAIModel()
    text = "LLaMA is a family of open-weight large language models developed by Meta. llama.cpp is a project that allows running LLaMA models efficiently on CPUs using C++."
    result = await model.run_task(text)
    print("🧠 ZAIModel Response:", result)


if __name__ == "__main__":
    asyncio.run(demo_zai_task())
