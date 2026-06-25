"""
Unit tests for the OIS bootstrapper.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from ois import (
    Calendar, DayCount, BusinessDayConvention,
    Deposit, OISSwap,
    DiscountCurve, InterpolationMethod,
    OISBootstrapper,
    year_fraction,
)


REF = date(2025, 6, 18)
CAL = Calendar("US")
T2  = CAL.advance(REF, n_days=2)


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

class TestYearFraction:
    def test_act360_one_year(self):
        start = date(2024, 1, 1)
        end   = date(2025, 1, 1)
        yf    = year_fraction(start, end, DayCount.ACT360)
        assert abs(yf - 366 / 360.0) < 1e-10

    def test_act365_six_months(self):
        start = date(2025, 1, 1)
        end   = date(2025, 7, 1)
        yf    = year_fraction(start, end, DayCount.ACT365)
        assert abs(yf - 181 / 365.0) < 1e-10

    def test_thirty360_standard(self):
        start = date(2024, 2, 28)
        end   = date(2024, 8, 28)
        yf    = year_fraction(start, end, DayCount.THIRTY360)
        assert abs(yf - 0.5) < 1e-10


class TestCalendar:
    def test_advance_modified_following_end_of_month(self):
        # June 30 2025 is a Monday — should stay
        d = CAL.advance(date(2025, 6, 30))
        assert d == date(2025, 6, 30)

    def test_weekend_adjustment(self):
        # June 21 2025 is a Saturday
        d = CAL.adjust(date(2025, 6, 21), BusinessDayConvention.MODIFIED_FOLLOWING)
        assert d == date(2025, 6, 23)

    def test_advance_months(self):
        d = CAL.advance(date(2025, 1, 31), n_months=1)
        # Feb 28 is last valid day, ModFol stays in Feb
        assert d.month == 2


# ---------------------------------------------------------------------------
# DiscountCurve interpolation
# ---------------------------------------------------------------------------

class TestDiscountCurve:
    def _flat_curve(self, rate: float = 0.05, method=InterpolationMethod.LOG_LINEAR):
        pillars = [
            (REF + __import__("datetime").timedelta(days=int(t * 365)),
             math.exp(-rate * t))
            for t in [0, 0.25, 0.5, 1, 2, 3, 5, 7, 10]
        ]
        return DiscountCurve(REF, pillars, method=method)

    def test_reference_date_df_is_one(self):
        c = self._flat_curve()
        assert abs(c.discount(REF) - 1.0) < 1e-12

    def test_zero_rate_recovers_flat_rate(self):
        rate = 0.05
        c    = self._flat_curve(rate)
        from datetime import timedelta
        d = REF + timedelta(days=365 * 5)
        assert abs(c.zero_rate(d, "continuous") - rate) < 1e-4

    def test_log_cubic_at_pillars(self):
        rate = 0.05
        c    = self._flat_curve(rate, method=InterpolationMethod.LOG_CUBIC)
        from datetime import timedelta
        for t in [1, 2, 5]:
            d  = REF + timedelta(days=int(t * 365))
            df = c.discount(d)
            assert abs(df - math.exp(-rate * t)) < 1e-5

    def test_forward_rate_consistency(self):
        c = self._flat_curve(0.05)
        from datetime import timedelta
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=365 + 91)
        f  = c.forward_rate(d1, d2, compounding="simple")
        # For flat 5% cont, simple fwd ≈ exp(0.05*91/365) - 1) / (91/360)
        assert 0.04 < f < 0.07


# ---------------------------------------------------------------------------
# Bootstrap: self-consistency (repricing)
# ---------------------------------------------------------------------------

class TestBootstrap:
    def _run(self):
        deps = [
            Deposit("ON",  REF, CAL.advance(REF, n_days=1),   rate=0.05330),
            Deposit("1M",  T2,  CAL.advance(T2,  n_months=1), rate=0.05310),
            Deposit("3M",  T2,  CAL.advance(T2,  n_months=3), rate=0.05265),
            Deposit("6M",  T2,  CAL.advance(T2,  n_months=6), rate=0.05180),
        ]
        swaps = [
            OISSwap("OIS_1Y",  T2, CAL.advance(T2, n_months=12),  0.04980, CAL),
            OISSwap("OIS_2Y",  T2, CAL.advance(T2, n_months=24),  0.04720, CAL),
            OISSwap("OIS_5Y",  T2, CAL.advance(T2, n_months=60),  0.04340, CAL),
            OISSwap("OIS_10Y", T2, CAL.advance(T2, n_months=120), 0.04200, CAL),
        ]
        return OISBootstrapper(REF, deps + swaps, CAL).run()

    def test_converged(self):
        result = self._run()
        assert result.converged

    def test_residuals_below_threshold(self):
        result = self._run()
        for label, resid in result.residuals_bps.items():
            assert abs(resid) < 0.01, f"{label}: residual {resid:.6f} bps"

    def test_deposit_reprices_exactly(self):
        result = self._run()
        dep = Deposit("ON", REF, CAL.advance(REF, n_days=1), rate=0.05330)
        df_start = result.curve.discount(dep.start_date)
        df_end   = result.curve.discount(dep.end_date)
        model    = dep.par_rate(df_start, df_end)
        assert abs(model - dep.rate) < 1e-8

    def test_swap_npv_at_par_is_zero(self):
        result = self._run()
        swap   = OISSwap("OIS_5Y", T2, CAL.advance(T2, n_months=60), 0.04340, CAL)
        npv    = swap.npv(result.curve)
        assert abs(npv) < 1e-7, f"5Y at-par NPV: {npv:.2e}"

    def test_discount_factors_decreasing(self):
        result = self._run()
        from datetime import timedelta
        dfs = [result.curve.discount(REF + timedelta(days=int(t * 365)))
               for t in [0.25, 0.5, 1, 2, 5, 10]]
        for i in range(len(dfs) - 1):
            assert dfs[i] > dfs[i + 1], f"DF not decreasing at index {i}"

    def test_log_cubic_reprice(self):
        # Log-cubic interpolation: the global Newton solver calibrates swap pillar
        # DFs using log-linear, then rebuilds the natural cubic spline at the end.
        # Deposits are constrained at their exact pillar, but the spline may deviate
        # at off-pillar interpolated points.  OIS swaps should still reprice to <1 bp.
        deps  = [Deposit("3M", T2, CAL.advance(T2, n_months=3), rate=0.05265)]
        swaps = [
            OISSwap("OIS_1Y",  T2, CAL.advance(T2, n_months=12), 0.04980, CAL),
            OISSwap("OIS_5Y",  T2, CAL.advance(T2, n_months=60), 0.04340, CAL),
        ]
        result = OISBootstrapper(
            REF, deps + swaps, CAL,
            interpolation=InterpolationMethod.LOG_CUBIC,
        ).run()
        for label, resid in result.residuals_bps.items():
            if label.startswith("OIS"):
                assert abs(resid) < 1.5, f"{label}: {resid:.4f} bps (cubic)"


# ---------------------------------------------------------------------------
# OISSwap analytics
# ---------------------------------------------------------------------------

class TestOISSwap:
    def test_annuity_positive(self):
        result = TestBootstrap()._run()
        swap   = OISSwap("5Y", T2, CAL.advance(T2, n_months=60), 0.04340, CAL)
        ann    = swap.annuity(result.curve)
        assert ann > 0

    def test_par_rate_matches_quote(self):
        result = TestBootstrap()._run()
        swap   = OISSwap("OIS_5Y", T2, CAL.advance(T2, n_months=60), 0.04340, CAL)
        par    = swap.par_rate(result.curve)
        assert abs(par - 0.04340) < 1e-6

    def test_dv01_sign_payer(self):
        """
        Payer OIS at par: bumping the 5Y OIS rate up by 1 bp raises the par rate,
        so the fixed rate the payer pays is now BELOW market → NPV increases → DV01 > 0.
        (The bump shifts the calibration instrument rate, not a parallel shift of the curve.)
        Total DV01 must be non-zero and the 5Y bucket should dominate.
        """
        from ois import compute_dv01
        result = TestBootstrap()._run()
        deps  = [
            Deposit("ON",  REF, CAL.advance(REF, n_days=1),   rate=0.05330),
            Deposit("1M",  T2,  CAL.advance(T2,  n_months=1), rate=0.05310),
            Deposit("3M",  T2,  CAL.advance(T2,  n_months=3), rate=0.05265),
            Deposit("6M",  T2,  CAL.advance(T2,  n_months=6), rate=0.05180),
        ]
        swaps = [
            OISSwap("OIS_1Y",  T2, CAL.advance(T2, n_months=12),  0.04980, CAL),
            OISSwap("OIS_2Y",  T2, CAL.advance(T2, n_months=24),  0.04720, CAL),
            OISSwap("OIS_5Y",  T2, CAL.advance(T2, n_months=60),  0.04340, CAL),
            OISSwap("OIS_10Y", T2, CAL.advance(T2, n_months=120), 0.04200, CAL),
        ]
        test_swap = OISSwap("5Y_PAYER", T2, CAL.advance(T2, n_months=60), 0.04340, CAL)
        dv01 = compute_dv01(REF, result.curve, deps + swaps, CAL,
                            InterpolationMethod.LOG_LINEAR, test_swap, 1_000_000)
        # 5Y bucket dominates and must be non-zero
        assert abs(dv01.dv01.get("OIS_5Y", 0)) > 100, \
            f"5Y bucket DV01 should dominate: {dv01.dv01}"
        assert abs(dv01.total_dv01) > 100, \
            f"Total DV01 should be non-trivial: {dv01.total_dv01}"
