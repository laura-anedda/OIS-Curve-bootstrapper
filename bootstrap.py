from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from .conventions import Calendar, DayCount, year_fraction
from .curve import DiscountCurve, InterpolationMethod
from .instruments import Deposit, OISSwap, FedFundsFuture, SOFRFuture


Instrument = Union[Deposit, OISSwap, FedFundsFuture, SOFRFuture]


@dataclass
class CalibrationResult:
    curve: DiscountCurve
    instruments: List[Instrument]
    residuals_bps: Dict[str, float]
    iterations: int
    converged: bool


class OISBootstrapper:
    """
    OIS discount curve bootstrapper with optional global Newton refinement.

    Pass 1 – Sequential analytic/Brent bootstrap (O(n)):
        Each pillar is solved analytically (deposits, OIS with adjacent pillars)
        or via Brent's method.  This gives a good initial guess.

    Pass 2 – Global Newton iteration (optional, default on):
        Treats the vector of log-DFs at pillar dates as unknowns and drives
        all instrument residuals to zero simultaneously.  Required when the
        calibration set has tenor gaps (e.g. 5Y, 7Y, 10Y but no 6Y) so that
        intermediate interpolated DFs are self-consistent.

    Parameters
    ----------
    reference_date : date
    instruments : list of Instrument
        Will be sorted by maturity internally.
    calendar : Calendar
    interpolation : str
        Interpolation method (log_linear | log_cubic | linear_zero).
    global_newton : bool
        Run global Newton refinement after sequential pass (default True).
    newton_tol_bps : float
        Convergence tolerance in basis points (default 1e-6).
    newton_maxiter : int
        Maximum Newton iterations (default 50).
    """

    _SOLVER_TOL      = 1e-12
    _SOLVER_MAXITER  = 200
    _RESIDUAL_TOL_BPS = 0.0001

    def __init__(
        self,
        reference_date: date,
        instruments: List[Instrument],
        calendar: Calendar,
        interpolation: str = InterpolationMethod.LOG_LINEAR,
        global_newton: bool = True,
        newton_tol_bps: float = 1e-6,
        newton_maxiter: int = 50,
    ):
        self.reference_date  = reference_date
        self.calendar        = calendar
        self.interpolation   = interpolation
        self.global_newton   = global_newton
        self.newton_tol      = newton_tol_bps / 1e4
        self.newton_maxiter  = newton_maxiter

        self.instruments = sorted(instruments, key=lambda x: x.maturity())

        self._curve = DiscountCurve(
            reference_date,
            [(reference_date, 1.0)],
            method=InterpolationMethod.LOG_LINEAR,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> CalibrationResult:
        # Pass 1: sequential bootstrap
        for inst in self.instruments:
            self._calibrate_instrument(inst)

        total_iter = len(self.instruments)

        # Pass 2: global Newton refinement
        if self.global_newton:
            newton_iter = self._global_newton()
            total_iter += newton_iter

        # Switch to final interpolation method
        if self.interpolation != InterpolationMethod.LOG_LINEAR:
            self._curve = DiscountCurve(
                self.reference_date,
                list(zip(self._curve._dates, self._curve._dfs)),
                method=self.interpolation,
            )

        residuals  = self._compute_residuals()
        converged  = all(abs(v) < self._RESIDUAL_TOL_BPS
                         for v in residuals.values())

        if not converged:
            warnings.warn(
                f"Bootstrap did not converge to {self._RESIDUAL_TOL_BPS} bps: "
                f"{ {k: round(v,4) for k,v in residuals.items() if abs(v) > self._RESIDUAL_TOL_BPS} }",
                RuntimeWarning,
            )

        return CalibrationResult(
            curve=self._curve,
            instruments=self.instruments,
            residuals_bps=residuals,
            iterations=total_iter,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Global Newton solver
    # ------------------------------------------------------------------

    def _global_newton(self) -> int:
        """
        Newton-Raphson iteration on all OIS-swap pillar DFs simultaneously.

        State vector x = [log DF(T_1), …, log DF(T_m)] where T_i are the
        maturity dates of OIS swaps (deposits are already exact).
        Jacobian is computed by finite difference (central, step 1e-7).
        """
        swap_instruments = [i for i in self.instruments if isinstance(i, OISSwap)]
        if not swap_instruments:
            return 0

        pillar_dates = [sw.maturity_date for sw in swap_instruments]

        def residuals_vec(log_dfs: np.ndarray) -> np.ndarray:
            for d, ldf in zip(pillar_dates, log_dfs):
                self._curve.update_pillar(d, math.exp(ldf))
            return np.array([
                sw.par_rate(self._curve) - sw.fixed_rate
                for sw in swap_instruments
            ])

        x = np.array([math.log(self._curve.discount(d)) for d in pillar_dates])

        step = 1e-7
        for it in range(self.newton_maxiter):
            r = residuals_vec(x)
            if np.max(np.abs(r)) < self.newton_tol:
                break

            # Finite-difference Jacobian
            J = np.zeros((len(x), len(x)))
            for j in range(len(x)):
                x_fwd = x.copy(); x_fwd[j] += step
                x_bwd = x.copy(); x_bwd[j] -= step
                J[:, j] = (residuals_vec(x_fwd) - residuals_vec(x_bwd)) / (2 * step)
            # Restore
            residuals_vec(x)

            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]

            # Damped step
            alpha = 1.0
            for _ in range(10):
                x_new = x + alpha * dx
                r_new = residuals_vec(x_new)
                if np.max(np.abs(r_new)) < np.max(np.abs(r)):
                    break
                alpha *= 0.5
            x = x + alpha * dx

        # Final update
        for d, ldf in zip(pillar_dates, x):
            self._curve.update_pillar(d, math.exp(ldf))

        return it + 1

    # ------------------------------------------------------------------
    # Per-instrument calibration (sequential pass)
    # ------------------------------------------------------------------

    def _calibrate_instrument(self, inst: Instrument):
        if isinstance(inst, Deposit):
            self._calibrate_deposit(inst)
        elif isinstance(inst, (FedFundsFuture, SOFRFuture)):
            self._calibrate_future(inst)
        elif isinstance(inst, OISSwap):
            self._calibrate_ois(inst)
        else:
            raise NotImplementedError(f"Unsupported instrument: {type(inst)}")

    def _calibrate_deposit(self, dep: Deposit):
        df_start = self._curve.discount(dep.start_date)
        dcf      = year_fraction(dep.start_date, dep.end_date, dep.day_count)
        df_end   = df_start / (1.0 + dep.rate * dcf)
        self._curve.update_pillar(dep.end_date, df_end)

    def _calibrate_future(self, fut: Union[FedFundsFuture, SOFRFuture]):
        r    = fut.implied_rate
        t0   = fut.delivery_start if isinstance(fut, FedFundsFuture) else fut.start_date
        t1   = fut.maturity()
        dcf  = year_fraction(t0, t1, fut.day_count)
        df_s = self._curve.discount(t0)
        df_e = df_s / (1.0 + r * dcf)
        self._curve.update_pillar(t1, df_e)

    def _calibrate_ois(self, swap: OISSwap):
        """
        Solve for DF(T_n) such that swap NPV = 0.

        The OIS par condition (fixed payer, notional 1):

            Σ_i [DF(T_{i-1}) - DF(T_i)]  =  S · Σ_i dcf_i · DF(T_i)

        The float leg telescopes to DF(T_0) - DF(T_n).  Isolating DF(T_n):

            DF(T_n) = [DF(T_0) - S · Σ_{i=1}^{n-1} dcf_i · DF(T_i)]
                      / [1 + S · dcf_n]

        T_0 is the swap effective date (first date in the fixed schedule).
        All intermediate pillars DF(T_1) … DF(T_{n-1}) are already fixed.
        """
        dates     = swap.fixed_schedule()
        day_count = swap.day_count
        S         = swap.fixed_rate

        num = self._curve.discount(dates[0])    # DF(T_0) = DF(effective date)

        for i in range(1, len(dates) - 1):
            dcf  = year_fraction(dates[i - 1], dates[i], day_count)
            df_i = self._curve.discount(dates[i])
            num -= S * dcf * df_i

        dcf_last    = year_fraction(dates[-2], dates[-1], day_count)
        df_maturity = num / (1.0 + S * dcf_last)

        if df_maturity <= 0:
            df_maturity = self._brent_ois(swap)

        self._curve.update_pillar(swap.maturity_date, df_maturity)

    def _brent_ois(self, swap: OISSwap) -> float:
        def objective(df_mat: float) -> float:
            self._curve.update_pillar(swap.maturity_date, df_mat)
            return swap.npv(self._curve)

        df_guess = self._curve.discount(swap.maturity_date)
        lo = max(df_guess * 0.1, 1e-6)
        hi = min(df_guess * 2.0, 1.0 - 1e-9)
        return brentq(objective, lo, hi, xtol=self._SOLVER_TOL,
                      maxiter=self._SOLVER_MAXITER)

    # ------------------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------------------

    def _compute_residuals(self) -> Dict[str, float]:
        residuals: Dict[str, float] = {}

        for inst in self.instruments:
            if isinstance(inst, Deposit):
                model = inst.par_rate(
                    self._curve.discount(inst.start_date),
                    self._curve.discount(inst.end_date),
                )
                residuals[inst.label] = (model - inst.rate) * 1e4

            elif isinstance(inst, (FedFundsFuture, SOFRFuture)):
                t0  = inst.delivery_start if isinstance(inst, FedFundsFuture) else inst.start_date
                t1  = inst.maturity()
                dcf = year_fraction(t0, t1, inst.day_count)
                df_s = self._curve.discount(t0)
                df_e = self._curve.discount(t1)
                model_rate = (df_s / df_e - 1.0) / dcf
                residuals[inst.label] = (model_rate - inst.implied_rate) * 1e4

            elif isinstance(inst, OISSwap):
                residuals[inst.label] = swap_residual_bps(inst, self._curve)

        return residuals


def swap_residual_bps(swap: OISSwap, curve: DiscountCurve) -> float:
    return (swap.par_rate(curve) - swap.fixed_rate) * 1e4

