from models.llama_model import LlamaModel

def test_real_llama_model():
    model = LlamaModel()

    user_input = "Explain quantum computing in simple terms."
    prompt = model.generate_prompt_from_template(user_input)
    print("\n📥 Prompt Sent to LLaMA:\n", prompt)

    response = model.generate_text(prompt=prompt)
    print("\n💬 LLaMA Response:\n", response)

if __name__ == "__main__":
    test_real_llama_model()
