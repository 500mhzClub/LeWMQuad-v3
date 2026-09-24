"""Type-exact canonical JSON comparisons for physical-claim evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""

    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (OverflowError, TypeError, ValueError):
        return False


def canonical_content_sha256(value: Mapping[str, Any], *, hash_field: str) -> str:
    content = dict(value)
    content.pop(hash_field, None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def canonical_content_sha256_valid(
    value: object,
    *,
    hash_field: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    stored = value.get(hash_field)
    if type(stored) is not str or len(stored) != 64 or any(
        char not in "0123456789abcdef" for char in stored
    ):
        return False
    try:
        return canonical_content_sha256(value, hash_field=hash_field) == stored
    except (OverflowError, TypeError, ValueError):
        return False


__all__ = [
    "canonical_content_sha256",
    "canonical_content_sha256_valid",
    "canonical_json_bytes",
    "canonical_json_equal",
]
