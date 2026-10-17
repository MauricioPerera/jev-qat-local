"""Bounded E2B preparation cache. The adapter's RLock serializes all access."""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time

MAX_BYTES = 64 * 1024 * 1024
TTL_SECONDS = 300
PREPARATION_VERSION = "e2b-media-v1"


def _size(value):
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_size(k) + _size(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(_size(v) for v in value)
    return total


def _fingerprint(path, limit):
    """Read a bounded, stable snapshot; hashes prevent stale same-path reuse."""
    def identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    with path.open("rb") as stream:
        import os
        before = os.fstat(stream.fileno())
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit:
        raise ValueError("Media file exceeds the configured size limit")
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise ValueError("Media changed while being read; retry with a stable file")
    return identity(after), hashlib.sha256(data).hexdigest()


class PreparedMediaCache:
    def __init__(self, max_bytes=MAX_BYTES, ttl_seconds=TTL_SECONDS):
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.preparations = 0

    def _remove(self, key):
        self.bytes -= self.entries.pop(key)["bytes"]

    def expire(self):
        now = time.monotonic()
        for key, entry in list(self.entries.items()):
            if now - entry["accessed"] >= self.ttl_seconds:
                self._remove(key)

    def release(self, path=None):
        for key in list(self.entries):
            if path is None or key[0] == str(Path(path).resolve()):
                self._remove(key)

    @contextmanager
    def content(self, *, media, modality, frames, adapter, loader, enabled):
        start = time.monotonic()
        self.expire()
        if modality == "text" or not enabled:
            with loader() as parts:
                yield parts, {"enabled": enabled, "hit": False, "items": len(parts),
                              "prepare_seconds": time.monotonic()-start, "cached_bytes": self.bytes}
            return
        path = adapter._media_path(media)  # Revalidate root, existence and size even on hits.
        fingerprint = _fingerprint(path, adapter.config["max_media_bytes"])
        key = (str(path), modality, frames if modality == "video" else None,
               adapter.config["audio_max_seconds"] if modality == "audio" else None,
               PREPARATION_VERSION, fingerprint)
        if key in self.entries:
            entry = self.entries[key]
            entry["accessed"] = time.monotonic()
            self.entries.move_to_end(key)
            parts = deepcopy(entry["parts"])
            self.hits += 1
            hit = True
        else:
            # Materialize before leaving the context: temporary WAV/JPEG files are removed.
            with loader() as loaded:
                parts = deepcopy(loaded)
            if _fingerprint(path, adapter.config["max_media_bytes"]) != fingerprint:
                raise ValueError("Media changed during preparation; retry with a stable file")
            self.preparations += 1
            hit = False
            self.release(path)  # Discard obsolete versions/settings of this same path.
            size = _size(parts) + _size(key)
            if size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    self._remove(next(iter(self.entries)))
                self.entries[key] = {"parts": deepcopy(parts), "bytes": size, "accessed": time.monotonic()}
                self.bytes += size
        yield parts, {"enabled": True, "hit": hit, "items": len(parts), "sha256": fingerprint[1],
                      "prepare_seconds": time.monotonic()-start, "cached_bytes": self.bytes,
                      "max_cached_bytes": self.max_bytes}
