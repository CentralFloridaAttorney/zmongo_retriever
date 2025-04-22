
class SafeResult:
    def __init__(self, data):
        self._data = data
    def model_dump(self):
        return self._data
    def __getitem__(self, item):
        return self._data[item]
    def get(self, key, default=None):
        return self._data.get(key, default)
    def __contains__(self, item):
        return item in self._data
    def __eq__(self, other):
        # Support test equality to dicts, lists, ints, etc.
        return self._data == other or (hasattr(other, "model_dump") and self._data == other.model_dump())
    def __repr__(self):
        return f"SafeResult({repr(self._data)})"