"""
Entry point: SOFR OIS curve bootstrapping from realistic market data.

Instrument set mirrors a standard USD OIS run:
  - O/N SOFR deposit (T+0 settle)
  - 1W, 2W, 1M, 2M, 3M, 6M, 9M deposits
  - 1Y – 30Y OIS swaps (annual fixed, quarterly float approximation)

Rates as of a representative snapshot (mid-market, not live).
"""

from __future__ import annotations

from datetime import date

from ois import (
    Calendar, DayCount, BusinessDayConvention,
    Deposit, OISSwap,
    OISBootstrapper, InterpolationMethod,
    print_calibration_report, print_curve_table, print_dv01_report, export_json,
    compute_dv01, parallel_shift_dv01,
)


def build_market_data(
    ref: date,
    cal: Calendar,
) -> tuple[list[Deposit], list[OISSwap]]:
    """
    Construct calibration instruments from market quotes.

    Deposit settle convention:
      O/N  → T+0 / T+1
      Term → T+2 business days (standard money-market)
    OIS swaps settle T+2.
    """
    T0  = ref                                   # O/N deposit starts today
    T2  = cal.advance(ref, n_days=2)            # standard spot date

    deposits = [
        Deposit("SOFR_ON",  T0,  cal.advance(T0,  n_days=1),   rate=0.05330),
        Deposit("SOFR_1W",  T2,  cal.advance(T2,  n_days=7),   rate=0.05325),
        Deposit("SOFR_2W",  T2,  cal.advance(T2,  n_days=14),  rate=0.05320),
        Deposit("SOFR_1M",  T2,  cal.advance(T2,  n_months=1), rate=0.05310),
        Deposit("SOFR_2M",  T2,  cal.advance(T2,  n_months=2), rate=0.05290),
        Deposit("SOFR_3M",  T2,  cal.advance(T2,  n_months=3), rate=0.05265),
        Deposit("SOFR_6M",  T2,  cal.advance(T2,  n_months=6), rate=0.05180),
        Deposit("SOFR_9M",  T2,  cal.advance(T2,  n_months=9), rate=0.05080),
    ]

    # OIS swap quotes: fixed rate (annual, Act/360) vs compounded SOFR
    ois_quotes = {
        "1Y":   0.04980,
        "2Y":   0.04720,
        "3Y":   0.04540,
        "4Y":   0.04420,
        "5Y":   0.04340,
        "7Y":   0.04250,
        "10Y":  0.04200,
        "12Y":  0.04185,
        "15Y":  0.04175,
        "20Y":  0.04165,
        "25Y":  0.04155,
        "30Y":  0.04140,
    }

    tenor_months = {
        "1Y": 12, "2Y": 24, "3Y": 36, "4Y": 48, "5Y": 60,
        "7Y": 84, "10Y": 120, "12Y": 144, "15Y": 180,
        "20Y": 240, "25Y": 300, "30Y": 360,
    }

    swaps = []
    for label, rate in ois_quotes.items():
        mat = cal.advance(T2, n_months=tenor_months[label])
        swaps.append(OISSwap(
            label          = f"OIS_{label}",
            effective_date = T2,
            maturity_date  = mat,
            fixed_rate     = rate,
            calendar       = cal,
            day_count      = DayCount.ACT360,
        ))

    return deposits, swaps


def main():
    ref = date(2025, 6, 18)
    cal = Calendar("US")

    deposits, swaps = build_market_data(ref, cal)
    instruments = deposits + swaps

    # ── Bootstrap ────────────────────────────────────────────────────────
    result = OISBootstrapper(
        reference_date = ref,
        instruments    = instruments,
        calendar       = cal,
        interpolation  = InterpolationMethod.LOG_LINEAR,
    ).run()

    # ── Reports ──────────────────────────────────────────────────────────
    print_calibration_report(result)
    print_curve_table(result.curve)

    # ── Swap pricing ─────────────────────────────────────────────────────
    # Price a 5Y payer OIS at current par — should show ~0 NPV
    T2 = cal.advance(ref, n_days=2)
    test_swap = OISSwap(
        label          = "5Y_PAYER_TEST",
        effective_date = T2,
        maturity_date  = cal.advance(T2, n_months=60),
        fixed_rate     = 0.04340,  # at-market
        calendar       = cal,
        day_count      = DayCount.ACT360,
    )

    notional = 10_000_000.0
    npv = test_swap.npv(result.curve) * notional
    par = test_swap.par_rate(result.curve) * 1e4

    print(f"  {'─'*50}")
    print(f"  Test position : 5Y payer OIS  N={notional:,.0f}")
    print(f"  Par rate      : {par:.4f} bps")
    print(f"  NPV (at-mkt)  : ${npv:,.4f}")
    print(f"  {'─'*50}\n")

    # ── Bucketed DV01 ────────────────────────────────────────────────────
    dv01 = compute_dv01(
        reference_date = ref,
        base_curve     = result.curve,
        instruments    = instruments,
        calendar       = cal,
        interpolation  = InterpolationMethod.LOG_LINEAR,
        swap           = test_swap,
        notional       = notional,
    )
    print_dv01_report(dv01, notional)

    pdv01 = parallel_shift_dv01(
        reference_date = ref,
        base_curve     = result.curve,
        instruments    = instruments,
        calendar       = cal,
        interpolation  = InterpolationMethod.LOG_LINEAR,
        swap           = test_swap,
        notional       = notional,
    )
    print(f"  Parallel-shift DV01 : ${pdv01:+,.2f} per bp\n")

    # ── Export ───────────────────────────────────────────────────────────
    export_json(result, "ois_curve.json")
    print("  Curve exported → ois_curve.json\n")


if __name__ == "__main__":
    main()
