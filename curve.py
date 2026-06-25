from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


class InterpolationMethod:
    LOG_LINEAR   = "log_linear"     # log-linear on DFs (industry standard)
    LOG_CUBIC    = "log_cubic"      # natural cubic spline on log(DF) — smoother fwd rates
    LINEAR_ZERO  = "linear_zero"    # linear on continuously-compounded zero rates


class DiscountCurve:
    """
    Discount factor curve defined on a set of pillar dates.

    Internally stores log(DF) on a time grid and interpolates.
    All times are ACT/365 year fractions from `reference_date`.

    Parameters
    ----------
    reference_date : date
        Valuation date (t = 0).
    pillars : list of (date, float)
        Sorted (date, discount_factor) pairs. Must include reference_date with DF=1.
    method : str
        Interpolation method (log_linear | log_cubic | linear_zero).
    """

    def __init__(
        self,
        reference_date: date,
        pillars: List[Tuple[date, float]],
        method: str = InterpolationMethod.LOG_LINEAR,
    ):
        self.reference_date = reference_date
        self.method = method

        pillars = sorted(pillars, key=lambda x: x[0])
        self._dates = [p[0] for p in pillars]
        self._dfs   = [p[1] for p in pillars]
        self._times = [self._t(d) for d in self._dates]
        self._log_dfs = [math.log(df) for df in self._dfs]

        self._build_interpolator()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _t(self, d: date) -> float:
        return (d - self.reference_date).days / 365.0

    def _build_interpolator(self):
        if self.method == InterpolationMethod.LOG_CUBIC:
            self._spline = CubicSpline(
                self._times, self._log_dfs,
                bc_type="natural",
            )
        # log_linear and linear_zero use numpy interp

    # ------------------------------------------------------------------
    # Core accessor
    # ------------------------------------------------------------------

    def discount(self, d: date) -> float:
        t = self._t(d)
        if t <= 0.0:
            return 1.0

        if self.method == InterpolationMethod.LOG_LINEAR:
            log_df = float(np.interp(t, self._times, self._log_dfs))
            return math.exp(log_df)

        if self.method == InterpolationMethod.LOG_CUBIC:
            return math.exp(float(self._spline(t)))

        if self.method == InterpolationMethod.LINEAR_ZERO:
            zeros = [-ldf / max(ti, 1e-9) for ldf, ti in
                     zip(self._log_dfs, self._times)]
            z = float(np.interp(t, self._times, zeros))
            return math.exp(-z * t)

        raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def zero_rate(
        self,
        d: date,
        compounding: str = "continuous",
        day_count_basis: float = 365.0,
    ) -> float:
        """
        Zero (spot) rate.

        compounding : "continuous" | "annual" | "semi" | "quarterly"
        """
        t = self._t(d)
        if t < 1e-9:
            return 0.0
        df = self.discount(d)
        if compounding == "continuous":
            return -math.log(df) / t
        if compounding == "annual":
            return df ** (-1.0 / t) - 1.0
        if compounding == "semi":
            return 2.0 * (df ** (-1.0 / (2.0 * t)) - 1.0)
        if compounding == "quarterly":
            return 4.0 * (df ** (-1.0 / (4.0 * t)) - 1.0)
        raise ValueError(f"Unknown compounding: {compounding}")

    def forward_rate(
        self,
        start: date,
        end: date,
        day_count_basis: float = 360.0,
        compounding: str = "simple",
    ) -> float:
        """
        Instantaneous or period forward rate between start and end.

        compounding : "simple" (money market) | "continuous"
        """
        df_s = self.discount(start)
        df_e = self.discount(end)
        if df_e < 1e-15:
            return 0.0
        accrual = (end - start).days / day_count_basis
        if accrual < 1e-9:
            return 0.0
        fwd_df = df_e / df_s
        if compounding == "simple":
            return (1.0 / fwd_df - 1.0) / accrual
        if compounding == "continuous":
            return -math.log(fwd_df) / accrual
        raise ValueError(f"Unknown compounding: {compounding}")

    def instantaneous_forward(self, d: date, bump_days: int = 1) -> float:
        """f(t) = -d(ln DF)/dt approximated by finite difference over bump_days."""
        from datetime import timedelta
        d1 = d
        d2 = d + timedelta(bump_days)
        df1 = self.discount(d1)
        df2 = self.discount(d2)
        dt  = bump_days / 365.0
        return -math.log(df2 / df1) / dt

    # ------------------------------------------------------------------
    # Curve update (used by bootstrapper during sequential calibration)
    # ------------------------------------------------------------------

    def add_pillar(self, d: date, df: float):
        """Append a new pillar and rebuild the interpolator."""
        t = self._t(d)
        # Insert sorted
        idx = next((i for i, ti in enumerate(self._times) if ti >= t), len(self._times))
        self._dates.insert(idx, d)
        self._dfs.insert(idx, df)
        self._times.insert(idx, t)
        self._log_dfs.insert(idx, math.log(df))
        self._build_interpolator()

    def update_pillar(self, d: date, df: float):
        """Update an existing pillar's discount factor and rebuild."""
        t = self._t(d)
        for i, ti in enumerate(self._times):
            if abs(ti - t) < 1e-9:
                self._dfs[i]     = df
                self._log_dfs[i] = math.log(df)
                self._build_interpolator()
                return
        self.add_pillar(d, df)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self, tenors_years: Optional[List[float]] = None) -> dict:
        """Serialise curve to dict for reporting."""
        from datetime import timedelta
        dates = (
            [self.reference_date + timedelta(days=int(t * 365))
             for t in tenors_years]
            if tenors_years
            else self._dates
        )
        rows = []
        for d in dates:
            rows.append({
                "date":        d.isoformat(),
                "t":           round(self._t(d), 6),
                "df":          round(self.discount(d), 8),
                "zero_cont":   round(self.zero_rate(d, "continuous") * 1e4, 4),   # bps
                "zero_annual": round(self.zero_rate(d, "annual") * 1e4, 4),
            })
        return {"reference_date": self.reference_date.isoformat(), "pillars": rows}
