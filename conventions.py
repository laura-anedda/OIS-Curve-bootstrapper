from __future__ import annotations

import calendar
from datetime import date, timedelta
from enum import Enum
from typing import List


class DayCount(Enum):
    ACT360   = "Act/360"
    ACT365   = "Act/365"
    ACT_ACT  = "Act/Act"
    THIRTY360 = "30/360"


class BusinessDayConvention(Enum):
    FOLLOWING          = "Following"
    MODIFIED_FOLLOWING = "ModifiedFollowing"
    PRECEDING          = "Preceding"
    UNADJUSTED         = "Unadjusted"


# ---------------------------------------------------------------------------
# Holiday calendars (minimal; extend with QuantLib or real feed in prod)
# ---------------------------------------------------------------------------

class Calendar:
    """Simplified TARGET / US calendars — weekends only for portability."""

    _US_HOLIDAYS_2024_2026: set[date] = {
        # New Year's
        date(2024, 1, 1), date(2025, 1, 1), date(2026, 1, 1),
        # MLK
        date(2024, 1, 15), date(2025, 1, 20), date(2026, 1, 19),
        # Presidents
        date(2024, 2, 19), date(2025, 2, 17), date(2026, 2, 16),
        # Memorial
        date(2024, 5, 27), date(2025, 5, 26), date(2026, 5, 25),
        # Juneteenth
        date(2024, 6, 19), date(2025, 6, 19), date(2026, 6, 19),
        # Independence
        date(2024, 7, 4), date(2025, 7, 4), date(2026, 7, 4),
        # Labor
        date(2024, 9, 2), date(2025, 9, 1), date(2026, 9, 7),
        # Thanksgiving
        date(2024, 11, 28), date(2025, 11, 27), date(2026, 11, 26),
        # Christmas
        date(2024, 12, 25), date(2025, 12, 25), date(2026, 12, 25),
    }

    _TARGET_HOLIDAYS: set[date] = {
        date(2024, 1, 1), date(2025, 1, 1), date(2026, 1, 1),
        date(2024, 3, 29), date(2025, 4, 18), date(2026, 4, 3),   # Good Friday
        date(2024, 4, 1), date(2025, 4, 21), date(2026, 4, 6),    # Easter Monday
        date(2024, 5, 1), date(2025, 5, 1), date(2026, 5, 1),     # Labour Day
        date(2024, 12, 25), date(2025, 12, 25), date(2026, 12, 25),
        date(2024, 12, 26), date(2025, 12, 26), date(2026, 12, 26),
    }

    def __init__(self, name: str = "US"):
        self.name = name
        self._holidays = (
            self._US_HOLIDAYS_2024_2026 if name == "US" else self._TARGET_HOLIDAYS
        )

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        if convention == BusinessDayConvention.UNADJUSTED or self.is_business_day(d):
            return d
        if convention == BusinessDayConvention.FOLLOWING:
            while not self.is_business_day(d):
                d += timedelta(1)
        elif convention == BusinessDayConvention.PRECEDING:
            while not self.is_business_day(d):
                d -= timedelta(1)
        elif convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            original_month = d.month
            candidate = d
            while not self.is_business_day(candidate):
                candidate += timedelta(1)
            if candidate.month != original_month:
                candidate = d
                while not self.is_business_day(candidate):
                    candidate -= timedelta(1)
            d = candidate
        return d

    def advance(self, d: date, n_days: int = 0, n_months: int = 0,
                convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING) -> date:
        if n_months:
            m = d.month + n_months
            y = d.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            max_day = calendar.monthrange(y, m)[1]
            d = date(y, m, min(d.day, max_day))
        if n_days:
            d += timedelta(n_days)
        return self.adjust(d, convention)

    def business_days_between(self, start: date, end: date) -> int:
        count = 0
        d = start
        while d < end:
            if self.is_business_day(d):
                count += 1
            d += timedelta(1)
        return count


# ---------------------------------------------------------------------------
# Day count fractions
# ---------------------------------------------------------------------------

def year_fraction(start: date, end: date, convention: DayCount = DayCount.ACT360) -> float:
    if convention == DayCount.ACT360:
        return (end - start).days / 360.0
    if convention == DayCount.ACT365:
        return (end - start).days / 365.0
    if convention == DayCount.ACT_ACT:
        # ICMA Act/Act: handle year boundary
        if start.year == end.year:
            return (end - start).days / (366.0 if calendar.isleap(start.year) else 365.0)
        frac = 0.0
        y = start.year
        cur = start
        while cur.year < end.year:
            yr_end = date(y + 1, 1, 1)
            yr_days = 366 if calendar.isleap(y) else 365
            frac += (yr_end - cur).days / yr_days
            cur = yr_end
            y += 1
        yr_days = 366 if calendar.isleap(end.year) else 365
        frac += (end - cur).days / yr_days
        return frac
    if convention == DayCount.THIRTY360:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year
        d1 = min(d1, 30)
        if d1 == 30:
            d2 = min(d2, 30)
        return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0
    raise ValueError(f"Unknown DayCount: {convention}")


def schedule(
    start: date,
    end: date,
    freq_months: int,
    calendar: Calendar,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    stub: str = "short_front",
) -> List[date]:
    """
    Generate a coupon schedule between start and end.

    Strategy: build backwards from end in steps of freq_months (preserving
    end-of-month anchoring, which is standard in swap markets).  Any short
    front stub — i.e. a period between start and the first regular coupon date
    that is shorter than (freq_months - 1) months — is merged into that first
    coupon period so the schedule always begins exactly at start.

    Result: sorted list of dates [start, c1, c2, …, end].
    """
    dates: List[date] = [end]
    d = end
    while True:
        prev = calendar.advance(d, n_months=-freq_months, convention=convention)
        if prev <= start:
            break
        dates.append(prev)
        d = prev
    dates.append(start)
    dates.reverse()

    # Collapse a spurious short front stub:
    # if dates[1] is less than (freq_months - 1) months after dates[0], remove it.
    if len(dates) > 2:
        stub_threshold_days = (freq_months - 1) * 28  # conservative lower bound
        if (dates[1] - dates[0]).days < stub_threshold_days:
            dates.pop(1)

    return dates
