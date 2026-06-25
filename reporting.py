from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

from .bootstrap import CalibrationResult
from .curve import DiscountCurve
from .sensitivity import BucketDV01


_SEP = "─" * 80


def print_calibration_report(result: CalibrationResult):
    print(f"\n{'═'*80}")
    print(f"  OIS CURVE CALIBRATION REPORT")
    print(f"  Reference date : {result.curve.reference_date.isoformat()}")
    print(f"  Instruments    : {len(result.instruments)}")
    print(f"  Interpolation  : {result.curve.method}")
    print(f"  Status         : {'✓ CONVERGED' if result.converged else '✗ NOT CONVERGED'}")
    print(f"{'═'*80}")

    print(f"\n  {'Label':<16}  {'Maturity':>10}  {'Market Rate':>12}  "
          f"{'Model Rate':>12}  {'Residual':>10}  {'DF':>12}")
    print(f"  {_SEP}")

    for inst in result.instruments:
        mat   = inst.maturity().isoformat()
        label = inst.label
        resid = result.residuals_bps.get(label, float("nan"))

        if hasattr(inst, "rate"):
            mkt = inst.rate * 1e4
        elif hasattr(inst, "implied_rate"):
            mkt = inst.implied_rate * 1e4
        elif hasattr(inst, "fixed_rate"):
            mkt = inst.fixed_rate * 1e4
        else:
            mkt = float("nan")

        mdl  = mkt - resid     # residual = model - market → model = market + residual... wait
        # correct: residual = (model_par - market) * 1e4 → model_par = market + resid/1e4
        mdl  = mkt + resid
        df   = result.curve.discount(inst.maturity())

        flag = "" if abs(resid) < 0.01 else " ◄"
        print(f"  {label:<16}  {mat:>10}  {mkt:>11.4f}b  "
              f"{mdl:>11.4f}b  {resid:>+9.4f}b  {df:>12.8f}{flag}")

    print(f"\n  Max residual: {max(abs(v) for v in result.residuals_bps.values()):.4f} bps\n")


def print_curve_table(
    curve: DiscountCurve,
    output_tenors: Optional[List[float]] = None,
):
    tenors = output_tenors or [
        1/52, 1/12, 2/12, 3/12, 6/12, 9/12,
        1, 2, 3, 4, 5, 7, 10, 15, 20, 30,
    ]
    from datetime import timedelta

    print(f"\n  OIS DISCOUNT CURVE  —  {curve.reference_date.isoformat()}")
    print(f"\n  {'Tenor':<8}  {'Date':>10}  {'DF':>12}  "
          f"{'Zero (cont, bps)':>17}  {'Zero (ann, bps)':>16}  {'Fwd 3M (bps)':>13}")
    print(f"  {_SEP}")

    for t in tenors:
        d  = curve.reference_date + timedelta(days=int(t * 365))
        df = curve.discount(d)
        z_cont = curve.zero_rate(d, "continuous") * 1e4
        z_ann  = curve.zero_rate(d, "annual")     * 1e4

        # 3M forward starting at d
        from datetime import timedelta as td
        d3m   = d + td(91)
        fwd3m = curve.forward_rate(d, d3m, compounding="simple") * 1e4

        label = _tenor_label(t)
        print(f"  {label:<8}  {d.isoformat():>10}  {df:>12.8f}  "
              f"{z_cont:>16.4f}b  {z_ann:>15.4f}b  {fwd3m:>12.4f}b")

    print()


def print_dv01_report(dv01: BucketDV01, notional: float = 1_000_000.0):
    print(f"\n  BUCKETED DV01  —  {dv01.position_label}  (notional {notional:,.0f})\n")
    print(f"  {'Bucket':<16}  {'DV01 ($)':>12}")
    print(f"  {'─'*32}")
    for label, val in dv01.dv01.items():
        bar  = "█" * max(0, int(abs(val) / max(abs(v) for v in dv01.dv01.values()) * 20))
        sign = "+" if val >= 0 else "-"
        print(f"  {label:<16}  {val:>+11.2f}  {bar}")
    print(f"  {'─'*32}")
    print(f"  {'Total DV01':<16}  {dv01.total_dv01:>+11.2f}\n")


def export_json(result: CalibrationResult, path: str):
    data = {
        "reference_date": result.curve.reference_date.isoformat(),
        "converged":      result.converged,
        "residuals_bps":  {k: round(v, 6) for k, v in result.residuals_bps.items()},
        "curve":          result.curve.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _tenor_label(t: float) -> str:
    if t < 1.0 / 11:
        return f"{int(round(t * 52))}W"
    if t < 1.0:
        return f"{int(round(t * 12))}M"
    return f"{int(round(t))}Y"
