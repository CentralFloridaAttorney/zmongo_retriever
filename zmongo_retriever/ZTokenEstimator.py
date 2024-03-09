from zmongo_retriever.data_tools import convert_object_to_json


class ZTokenEstimator:
    def __init__(self):
        self.char_groups = [
            ("  ", 0.081),
            ("NORabcdefghilnopqrstuvy ", 0.202),
            ("CHLMPQSTUVfkmspwx", 0.237),
            ("-.ABDEFGIKWY_\\r\tz{ü", 0.304),
            ("!$&(/;=JX`j\n}ö", 0.416),
            ("\"#%)*+56789<>?@Z[\\]^|§«äç’", 0.479),
            (",01234:~Üß", 0.658),
        ]
        self.meta_tag_token_value = 5

    def estimate_tokens_from_text(self, text):
        estimate = 0
        while text:
            if text.startswith('<meta>') and '</meta>' in text:
                meta_end_idx = text.find('</meta>') + len('</meta>')
                estimate += self.meta_tag_token_value
                text = text[meta_end_idx:]
            else:
                char = text[0]
                text = text[1:]
                for group, value in self.char_groups:
                    if char in group or ord(char) > 255:
                        estimate += value
                        break
                else:
                    estimate += 0.98
        return estimate

    def estimate_tokens(self, item_to_estimate):
        item = convert_object_to_json(item_to_estimate)
        if isinstance(item, dict):
            # Process each value in the dictionary recursively
            return sum(self.estimate_tokens(value) for value in item.values())
        elif isinstance(item, list):
            # Process each item in the list recursively
            return sum(self.estimate_tokens(element) for element in item)
        elif isinstance(item, str):
            return self.estimate_tokens_from_text(item)
        else:
            # Non-string and non-collection items are not processed
            return 0

    @staticmethod
    def main():
        estimator = ZTokenEstimator()
        items = [
            "This is a simple test.",
            "<meta>This is some metadata</meta>This is the main content.",
            {"metadata": {"source": "mongodb", "database_name": "case_graph", "collection_name": "zcases", "document_id": "65d8df9fb347eec24ffa7671", "page_content_field": "opinion"},
             "page_content": "Justice White delivered the opinion of the Court..."},
            ["This is a list item.", {"nested_dict": "This is nested."}],
            "function calculate() { return 42; }",
            "<html><body>Hello, world!</body></html>",
            "# Markdown Header\n\nThis is a markdown file.",
        ]

        for item in items:
            est_tokens = estimator.estimate_tokens(item)
            preview = str(item)[:30] + "..." if isinstance(item, str) else str(type(item))
            print(f"Item: {preview} Estimated tokens: {est_tokens}")


if __name__ == "__main__":
    ZTokenEstimator.main()
