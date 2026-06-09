from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class NormalizedTime:
    time_start: datetime | None
    time_end: datetime | None
    granularity: str
    source: str


DATE_RE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
MONTH_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b",
    re.I,
)
WEEKDAY_RE = re.compile(r"\b(this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class TemporalNormalizer:
    def normalize(self, text: str, session_time: datetime) -> NormalizedTime:
        if session_time.tzinfo is None:
            session_time = session_time.replace(tzinfo=timezone.utc)
        tz = session_time.tzinfo
        lower = text.lower()
        match = DATE_RE.search(text)
        if match:
            year, month, day = map(int, match.groups())
            return _explicit_day(year, month, day, tz, "explicit")
        month_match = MONTH_RE.search(text)
        if month_match:
            month_text, day_text, year_text = month_match.groups()
            year = int(year_text) if year_text else session_time.year
            return _explicit_day(year, MONTHS[month_text.lower()], int(day_text), tz, "explicit")
        if "yesterday" in lower:
            start = _day_start(session_time - timedelta(days=1))
            return NormalizedTime(start, start + timedelta(days=1), "day", "relative_resolved")
        if "today" in lower:
            start = _day_start(session_time)
            return NormalizedTime(start, start + timedelta(days=1), "day", "relative_resolved")
        if "tomorrow" in lower:
            start = _day_start(session_time + timedelta(days=1))
            return NormalizedTime(start, start + timedelta(days=1), "day", "relative_resolved")
        if "last week" in lower:
            start = _day_start(session_time - timedelta(days=session_time.weekday() + 7))
            return NormalizedTime(start, start + timedelta(days=7), "week", "relative_resolved")
        if "this week" in lower:
            start = _day_start(session_time - timedelta(days=session_time.weekday()))
            return NormalizedTime(start, start + timedelta(days=7), "week", "relative_resolved")
        if "next week" in lower:
            start = _day_start(session_time - timedelta(days=session_time.weekday()) + timedelta(days=7))
            return NormalizedTime(start, start + timedelta(days=7), "week", "relative_resolved")
        if "last month" in lower:
            start = _month_start(_add_months(session_time, -1))
            return NormalizedTime(start, _add_months(start, 1), "month", "relative_resolved")
        if "this month" in lower:
            start = _month_start(session_time)
            return NormalizedTime(start, _add_months(start, 1), "month", "relative_resolved")
        if "next month" in lower:
            start = _month_start(_add_months(session_time, 1))
            return NormalizedTime(start, _add_months(start, 1), "month", "relative_resolved")
        weekday_match = WEEKDAY_RE.search(text)
        if weekday_match:
            modifier, weekday = weekday_match.groups()
            start = _weekday_start(session_time, modifier.lower(), weekday.lower())
            return NormalizedTime(start, start + timedelta(days=1), "day", "relative_resolved")
        return NormalizedTime(None, None, "unknown", "unknown")


def _day_start(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _weekday_start(session_time: datetime, modifier: str, weekday: str) -> datetime:
    target = WEEKDAYS[weekday]
    week_start = _day_start(session_time - timedelta(days=session_time.weekday()))
    if modifier == "this":
        return week_start + timedelta(days=target)
    days_ahead = (target - session_time.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return _day_start(session_time + timedelta(days=days_ahead))


def _explicit_day(year: int, month: int, day: int, tz, source: str) -> NormalizedTime:
    try:
        start = datetime(year, month, day, tzinfo=tz)
    except ValueError:
        return NormalizedTime(None, None, "unknown", "unknown")
    return NormalizedTime(start, start + timedelta(days=1), "day", source)
