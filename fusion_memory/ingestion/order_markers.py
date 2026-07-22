from __future__ import annotations

import re


ORDER_RE = re.compile(r"\b(after|before)\s+(?:the\s+)?(.+?)(?:,|\.|;|\bthen\b|\bi\s+|\bwe\s+|$)", re.I)


def _explicit_order_mentions(text: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    for match in ORDER_RE.finditer(text):
        direction = match.group(1).lower()
        target = match.group(2).strip()
        if target:
            mentions.append((target, direction))
    return mentions
