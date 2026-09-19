"""
Tiny thread-safe TTL cache.

Every panel in the dashboard used to re-download 600 days of history and
re-run the model on every click. Caching is the single biggest latency win
for everything except the LLM.
"""
import threading
import time

_LOCK = threading.Lock()
_STORE = {}


def get(key):
    """Return the cached value, or None if missing/expired."""
    with _LOCK:
        entry = _STORE.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            _STORE.pop(key, None)
            return None
        return value


def put(key, value, ttl_seconds):
    """Store a value for ttl_seconds."""
    with _LOCK:
        _STORE[key] = (time.time() + ttl_seconds, value)


def clear():
    with _LOCK:
        _STORE.clear()


def stats():
    with _LOCK:
        now = time.time()
        live = sum(1 for exp, _ in _STORE.values() if exp > now)
        return {"entries": len(_STORE), "live": live}
