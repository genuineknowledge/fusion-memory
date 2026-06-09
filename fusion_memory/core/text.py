from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", text):
        value = match.group(0)
        if value.lower() not in {"user", "assistant", "agent"}:
            entities.append(value)
    seen: set[str] = set()
    out: list[str] = []
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            out.append(entity)
    return out


def cosine_sparse(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def keyword_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    return jaccard(q, t)


def compact_summary(text: str, limit: int = 360) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."

