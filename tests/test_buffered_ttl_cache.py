
import asyncio
import time
import types
import pytest

from buffered_ttl_cache import BufferedAsyncTTLCache

pytestmark = pytest.mark.asyncio


async def test_set_and_get_basic():
    cache = BufferedAsyncTTLCache(ttl=1)
    await cache.set("a", 123)
    assert await cache.get("a") == 123
    assert await cache.get("missing", default="x") == "x"


async def test_ttl_expiration_and_cleanup():
    cache = BufferedAsyncTTLCache(ttl=0.05)  # 50 ms
    await cache.set("k", "v")
    assert await cache.get("k") == "v"
    await asyncio.sleep(0.08)  # sleep past TTL with margin
    # expired path returns default & evicts
    assert await cache.get("k", default=None) is None
    # after cleanup, __len__ should be 0
    await cache.cleanup()
    assert len(cache) == 0
    assert await cache.async_len() == 0


async def test_delete_and_clear():
    cache = BufferedAsyncTTLCache(ttl=1)
    await cache.set("a", 1)
    await cache.set("b", 2)
    assert await cache.get("a") == 1
    assert await cache.get("b") == 2

    await cache.delete("a")
    assert await cache.get("a") is None
    assert await cache.get("b") == 2

    await cache.clear()
    assert await cache.get("b") is None
    assert await cache.async_len() == 0
    assert await cache.buffered_count() == 0


async def test_contains_and_len_respect_ttl():
    cache = BufferedAsyncTTLCache(ttl=0.05)
    await cache.set("x", "y")
    # immediately contained
    assert "x" in cache
    assert len(cache) == 1
    await asyncio.sleep(0.07)
    # contains uses current time; should consider expired as not present
    assert "x" not in cache
    # __len__ counts only non-expired
    assert len(cache) == 0


async def test_async_contains_and_async_len():
    cache = BufferedAsyncTTLCache(ttl=1)
    await cache.set("x", 9)
    assert await cache.async_contains("x") is True
    assert await cache.async_contains("nope") is False
    assert await cache.async_len() == 1


async def test_items_iterates_key_values():
    cache = BufferedAsyncTTLCache(ttl=1)
    await cache.set("a", 1)
    await cache.set("b", 2)
    seen = {}
    async for k, v in cache.items():
        seen[k] = v
    assert seen == {"a": 1, "b": 2}


async def test_buffer_only_accumulates_and_does_not_flush_automatically():
    cache = BufferedAsyncTTLCache(ttl=1)
    await cache.set("k1", "v1", buffer_only=True)
    await cache.set("k2", "v2", buffer_only=True)
    # buffer should grow, cache should remain empty
    assert await cache.buffered_count() == 2
    assert await cache.async_len() == 0


async def test_flush_invokes_flush_fn_and_clears_buffer():
    received_batches = []

    async def fake_flush_fn(batch):
        # simulate I/O latency a little
        await asyncio.sleep(0.01)
        received_batches.append(batch)

    cache = BufferedAsyncTTLCache(ttl=1, flush_fn=fake_flush_fn)
    await cache.set("k1", "v1", buffer_only=True)
    await cache.set("k2", "v2", buffer_only=True)

    assert await cache.buffered_count() == 2
    await cache.flush()
    # flush clears buffer and calls the function once with all pairs
    assert await cache.buffered_count() == 0
    assert received_batches == [[("k1", "v1"), ("k2", "v2")]]


async def test_flush_is_noop_without_flush_fn():
    cache = BufferedAsyncTTLCache(ttl=1, flush_fn=None)
    await cache.set("k", "v", buffer_only=True)
    assert await cache.buffered_count() == 1
    # should not raise and should leave buffer untouched
    await cache.flush()
    assert await cache.buffered_count() == 1


async def test_concurrent_sets_and_gets_are_safe():
    cache = BufferedAsyncTTLCache(ttl=1)

    async def writer(n):
        for i in range(n):
            await cache.set(f"k{i}", i)
            # tiny context switch
            await asyncio.sleep(0)

    async def reader(n):
        # Try to read while writes are happening; ensure no exceptions
        total = 0
        for i in range(n):
            _ = await cache.get(f"k{i}")
            await asyncio.sleep(0)
            total += 1
        return total

    n = 100
    results = await asyncio.gather(writer(n), reader(n))
    # writer returns None, reader returns total loops
    assert results[1] == n
    # after all, many keys should exist (no strict guarantee all written before TTL expires)
    assert (await cache.async_len()) > 0


async def test_cleanup_removes_only_expired():
    cache = BufferedAsyncTTLCache(ttl=0.05)
    await cache.set("short", "s")
    await asyncio.sleep(0.02)
    await cache.set("later", "l")
    # sleep enough to expire "short" but not "later"
    await asyncio.sleep(0.04)  # total ~0.06 since first set

    await cache.cleanup()
    assert await cache.get("short") is None
    assert await cache.get("later") == "l"
