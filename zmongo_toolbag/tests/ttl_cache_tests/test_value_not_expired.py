import time
import pytest

from zai_toolbag.ttl_cache import TTLCache


def test_ttlcache_value_not_expired():
    # Create a TTLCache instance with a TTL of 5 seconds.
    cache = TTLCache(default_ttl=5)

    # Set a key "foo" with value "bar"
    cache.set("foo", "bar")

    # Wait for 2 seconds, which is less than the TTL.
    time.sleep(2)

    # Since 2 < 5, the key should not be expired.
    value = cache.get("foo")
    assert value == "bar", f"Expected 'bar', but got {value}"


if __name__ == "__main__":
    pytest.main([__file__])
