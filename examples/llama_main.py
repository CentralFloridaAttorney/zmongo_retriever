from zmongo_retriever.examples.llama_model import LlamaModel


def main():
    llama_model = LlamaModel()

    user_input = (
        "Write a Dungeons & Dragons encounter using D20 rules. "
        "Include full descriptive text for the dungeon master to read when running the encounter. "
        "This is for new dungeon masters. The adventurers awake from a drunken slumber in the corner of a tavern."
    )

    prompt = llama_model.generate_prompt_from_template(user_input)

    output_text = llama_model.generate_text(
        prompt=prompt,
        max_tokens=3000,  # careful with this value depending on your model!
    )

    print("Generated Text:\n")
    print(output_text)


if __name__ == "__main__":
    main()
