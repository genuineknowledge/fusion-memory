from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class TaxonomyEntry:
    label: str
    aliases: list[str]
    tags: list[str]
    language: str = "unknown"


@lru_cache(maxsize=1)
def load_default_taxonomy() -> list[TaxonomyEntry]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_taxonomy.json"
    raw_entries = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        TaxonomyEntry(
            label=str(entry["label"]),
            aliases=[str(alias) for alias in entry.get("aliases", [])],
            tags=[str(tag) for tag in entry.get("tags", [])],
            language=str(entry.get("language", "unknown")),
        )
        for entry in raw_entries
    ]


def taxonomy_alias_hits(text: str, entries: list[TaxonomyEntry] | None = None) -> set[str]:
    haystack = str(text or "")
    selected_entries = entries if entries is not None else load_default_taxonomy()
    hits: set[str] = set()
    for entry in selected_entries:
        for alias in entry.aliases:
            if _alias_present(haystack, alias):
                hits.add(entry.label)
                break
    return hits


def _alias_present(text: str, alias: str) -> bool:
    normalized_alias = alias.strip()
    if not normalized_alias:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
