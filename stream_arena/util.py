"""Small shared helpers: canonical hashing, JSON/JSONL I/O, run-artifact records, credential scrubbing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def digest(value) -> str:
    """sha256 of the canonical JSON encoding (sorted keys, UTF-8, default separators); the hash every frozen
    plan, session and oracle binding in the benchmark uses."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value, *, indent=2) -> None:
    """Atomic write (temp file + rename), so a reader never sees half a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temp.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def private_json(path, value) -> None:
    """Run artifact: owner-only file (0600) in an owner-only directory, written atomically."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("a run artifact cannot be a symbolic link")
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


SENSITIVE = {"api_key", "authorization", "token", "client_secret", "secret", "rtc", "headers", "access_token",
             "app_secret", "dashscope_api_key"}


def scrub(value):
    """Drop credential-bearing keys and mask credentials, inline media and URLs in text before it is recorded."""
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items() if key.lower() not in SENSITIVE}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"sk-[A-Za-z0-9_-]+", "[credential omitted]", value)
        value = re.sub(r"(?i)(?:Bearer|Token)\s+\S+", "[credential omitted]", value)
        value = re.sub(r"data:[^\s]+", "[inline media omitted]", value)
        value = re.sub(r"(?i)(?:https?|wss?|rtmp)://[^\s\"<>]+", "[URL omitted]", value)
        return value[:4000]
    return value
