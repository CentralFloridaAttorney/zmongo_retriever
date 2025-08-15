import unittest
import time
from BAK.utils import TTLCache


class TestTTLCache(unittest.TestCase):
    def setUp(self):
        self.cache = TTLCache(ttl=1)  # 1 second TTL for quick expiry testing

    def test_set_and_get(self):
        self.cache.set("foo", "bar")
        self.assertEqual(self.cache.get("foo"), "bar")

    def test_expiration(self):
        self.cache.set("temp", "value", ttl=1)
        time.sleep(1.1)
        self.assertIsNone(self.cache.get("temp"))

    def test_get_with_default(self):
        self.assertEqual(self.cache.get("nonexistent", default="missing"), "missing")

    def test_delete(self):
        self.cache.set("delete_me", "bye")
        self.cache.delete("delete_me")
        self.assertIsNone(self.cache.get("delete_me"))

    def test_clear(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)

    def test_cleanup(self):
        self.cache.set("will_expire", "soon", ttl=1)
        self.cache.set("will_stay", "longer", ttl=10)
        time.sleep(1.1)
        self.cache.cleanup()
        self.assertIsNone(self.cache.get("will_expire"))
        self.assertEqual(self.cache.get("will_stay"), "longer")

    def test_len_excludes_expired(self):
        self.cache.set("short", "life", ttl=1)
        self.cache.set("long", "life", ttl=5)
        time.sleep(1.1)
        self.assertEqual(len(self.cache), 1)

    def test_contains(self):
        self.cache.set("present", "yes")
        self.assertIn("present", self.cache)
        time.sleep(1.1)
        self.assertNotIn("present", self.cache)

    def test_items(self):
        self.cache.set("x", 10)
        self.cache.set("y", 20)
        items = list(self.cache.items())
        self.assertEqual(len(items), 2)
        self.assertIn(("x", 10), items)
        self.assertIn(("y", 20), items)


if __name__ == "__main__":
    unittest.main()
