import difflib


class DiffUtils:
    @staticmethod
    def binary_diff(a, b):
        diff = difflib.ndiff(a, b)
        return '\n'.join(diff)

    @staticmethod
    def binary_safe_diff(a, b):
        a = a.encode('utf-8')
        b = b.encode('utf-8')
        diff = list(difflib.ndiff(a, b))
        return '\n'.join([line.decode('utf-8') for line in diff])

    @staticmethod
    def create_patch(a, b):
        diff = difflib.unified_diff(a.splitlines(), b.splitlines())
        return '\n'.join(diff)

    @staticmethod
    def apply_patch(a, patch):
        patch_lines = patch.splitlines()
        result = list(difflib.restore(patch_lines, 2))
        return '\n'.join(result)

    @staticmethod
    def string_diff(a, b):
        diff = difflib.ndiff(a, b)
        return '\n'.join(diff)

    @staticmethod
    def unicode_diff(a, b):
        diff = difflib.ndiff(a, b)
        return '\n'.join(diff)

    @staticmethod
    def bytes_match(a, b):
        if a == b:
            return "Bytes match"
        else:
            return "Bytes do not match"

    @staticmethod
    def unicode_match(a, b):
        if a == b:
            return "Strings match"
        else:
            return "Strings do not match"
