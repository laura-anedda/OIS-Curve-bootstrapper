from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional

from .conventions import (
    Calendar, DayCount, BusinessDayConvention,
    year_fraction, schedule,
)


class InstrumentType(Enum):
    DEPOSIT   = "Deposit"
    OIS_SWAP  = "OIS_Swap"
    FED_FUNDS_FUTURE = "FedFundsFuture"
    SOFR_FUTURE      = "SOFRFuture"


@dataclass
class CashFlow:
    payment_date: date
    amount: float          # as fraction of notional
    dcf: float             # day-count fraction for display / accrual


@dataclass
class Deposit:
    """Overnight or term deposit fixing the short end."""
    type: InstrumentType = field(default=InstrumentType.DEPOSIT, init=False)
    label: str
    start_date: date
    end_date: date
    rate: float                              # quoted rate (decimal)
    day_count: DayCount = DayCount.ACT360
    notional: float = 1.0

    def maturity(self) -> date:
        return self.end_date

    def implied_discount_factor(self) -> float:
        """DF from start to end using simple compounding (money-market convention)."""
        dcf = year_fraction(self.start_date, self.end_date, self.day_count)
        return 1.0 / (1.0 + self.rate * dcf)

    def par_rate(self, df_start: float, df_end: float) -> float:
        dcf = year_fraction(self.start_date, self.end_date, self.day_count)
        return (df_start / df_end - 1.0) / dcf


@dataclass
class OISSwap:
    """
    Fixed-vs-compounded-overnight swap (SOFR OIS, €STR OIS, SONIA OIS).

    Convention: fixed leg pays annually (Act/360); floating leg resets daily,
    compounds, pays at same freq as fixed. Settlement T+2 business days.
    """
    type: InstrumentType = field(default=InstrumentType.OIS_SWAP, init=False)
    label: str
    effective_date: date
    maturity_date: date
    fixed_rate: float                        # quoted coupon (decimal)
    calendar: Calendar
    fixed_freq_months: int = 12             # annual by default
    float_freq_months: int = 12
    day_count: DayCount = DayCount.ACT360
    fixed_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    notional: float = 1.0

    def fixed_schedule(self) -> List[date]:
        return schedule(
            self.effective_date, self.maturity_date,
            self.fixed_freq_months, self.calendar, self.fixed_convention,
        )

    def fixed_cash_flows(self) -> List[CashFlow]:
        dates = self.fixed_schedule()
        cfs: List[CashFlow] = []
        for i in range(1, len(dates)):
            dcf = year_fraction(dates[i - 1], dates[i], self.day_count)
            cfs.append(CashFlow(
                payment_date=dates[i],
                amount=self.fixed_rate * dcf,
                dcf=dcf,
            ))
        # notional repayment absorbed into floating; no bullet on fixed leg
        return cfs

    def maturity(self) -> date:
        return self.maturity_date

    def annuity(self, curve: "DiscountCurve") -> float:
        """PV01 (annuity): Σ dcf_i * DF(t_i)."""
        dates = self.fixed_schedule()
        a = 0.0
        for i in range(1, len(dates)):
            dcf = year_fraction(dates[i - 1], dates[i], self.day_count)
            a  += dcf * curve.discount(dates[i])
        return a

    def float_npv(self, curve: "DiscountCurve") -> float:
        """
        For a compounded OIS swap, the floating NPV telescopes to:
            Σ_i [DF(T_{i-1}) - DF(T_i)]  =  DF(T_0) - DF(T_n)
        where the sum is over fixed-schedule payment periods.
        This is exact under the daily-compounding approximation standard in
        OIS bootstrapping (see O'Kane 2008, §6).
        """
        dates = self.fixed_schedule()
        return curve.discount(dates[0]) - curve.discount(dates[-1])

    def npv(self, curve: "DiscountCurve") -> float:
        """NPV from fixed-payer perspective: float_npv - fixed_rate * annuity."""
        return self.float_npv(curve) - self.fixed_rate * self.annuity(curve)

    def par_rate(self, curve: "DiscountCurve") -> float:
        ann = self.annuity(curve)
        if ann < 1e-12:
            return 0.0
        return self.float_npv(curve) / ann


@dataclass
class FedFundsFuture:
    """
    30-Day Fed Funds Future. Settlement price = 100 - avg(FF rate over delivery month).
    Convexity adjustment is applied externally and baked into `implied_rate`.
    """
    type: InstrumentType = field(default=InstrumentType.FED_FUNDS_FUTURE, init=False)
    label: str
    delivery_start: date     # first calendar day of delivery month
    delivery_end: date       # last calendar day of delivery month
    price: float             # settlement price (e.g. 94.75)
    convexity_adj_bps: float = 0.0   # desk-supplied convexity adjustment
    day_count: DayCount = DayCount.ACT360

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0 - self.convexity_adj_bps / 1e4

    def maturity(self) -> date:
        return self.delivery_end

    def mid_date(self) -> date:
        """Effective date of the future: mid-point of delivery month."""
        delta = (self.delivery_end - self.delivery_start).days // 2
        from datetime import timedelta
        return self.delivery_start + timedelta(delta)


@dataclass
class SOFRFuture:
    """
    3-Month SOFR Future (CME). Implies 3M compounded SOFR rate.
    Convexity adjustment baked in externally.
    """
    type: InstrumentType = field(default=InstrumentType.SOFR_FUTURE, init=False)
    label: str
    start_date: date
    end_date: date
    price: float
    convexity_adj_bps: float = 0.0
    day_count: DayCount = DayCount.ACT360

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0 - self.convexity_adj_bps / 1e4

    def maturity(self) -> date:
        return self.end_date
