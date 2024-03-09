class Document:
    def __init__(self, page_content, this_metadata=None):
        self.page_content = page_content
        self.metadata = this_metadata if this_metadata else {}