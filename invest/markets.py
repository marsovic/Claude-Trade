"""US market calendar (NYSE full-day holidays) and split-ratio recognition.

Holiday rules (NYSE): New Year's Day (Sunday -> Monday; a Saturday New Year is
not observed), Martin Luther King Jr. Day, Washington's Birthday, Good Friday,
Memorial Day, Juneteenth (since 2022), Independence Day, Labor Day,
Thanksgiving and Christmas (Saturday -> Friday, Sunday -> Monday). One-off
closures (national mourning days...) are not known in advance.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@cache
def nyse_holidays(year: int) -> frozenset[date]:
    days = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        days.add(new_year + timedelta(days=1))
    elif new_year.weekday() < 5:
        days.add(new_year)
    days.add(_nth_weekday(year, 1, 0, 3))  # MLK
    days.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    days.add(_easter(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.add(_observed(date(year, 7, 4)))
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))
    return frozenset(days)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def holidays_between(start: date, end: date) -> list[date]:
    out = []
    for year in range(start.year, end.year + 1):
        out.extend(h for h in nyse_holidays(year) if start <= h <= end)
    return sorted(out)


def sessions_until(day: date, report: date) -> int:
    """Trading sessions after ``day`` up to and including ``report`` (0 if past)."""
    import numpy as np

    start = day + timedelta(days=1)
    if report < start:
        return 0
    hol = holidays_between(start, report)
    return int(np.busday_count(start, report + timedelta(days=1), holidays=hol))


_SPLITS = sorted({float(n) for n in range(2, 51)} | {1 / n for n in range(2, 51)}
                 | {1.5, 2.5, 4 / 3, 5 / 3, 2 / 3, 0.4})


def split_ratio(r: float | None) -> float | None:
    """The split ratio (2, 3, 3/2, 10, 1/10...) if ``r`` is within 1 % of one."""
    if r is None or r <= 0 or 0.75 < r < 1.4:
        return None
    best = min(_SPLITS, key=lambda x: abs(x - r) / x)
    return best if abs(best - r) / best <= 0.01 else None
