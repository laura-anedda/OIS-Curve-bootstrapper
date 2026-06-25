# OIS Curve Bootstrapper

Bootstraps a SOFR OIS discount curve from market instruments using a
two-pass algorithm: sequential analytic calibration followed by global
Newton-Raphson refinement for full curve consistency.

---

## Architecture

```
ois/
├── conventions.py   Calendar, day-count fractions, coupon schedule generation
├── instruments.py   Deposit, OISSwap, FedFundsFuture, SOFRFuture
├── curve.py         DiscountCurve (log-linear / log-cubic / linear-zero interpolation)
├── bootstrap.py     OISBootstrapper: sequential pass + global Newton solver
├── sensitivity.py   Bucketed DV01, parallel-shift DV01 (bump-and-reprice)
└── reporting.py     Calibration report, curve table, DV01 grid, JSON export
```

---

## Methodology

### Instrument set

| Segment | Instruments |
|---------|-------------|
| O/N     | SOFR overnight deposit (T+0 settle) |
| Short end | 1W – 9M term deposits (T+2 settle, Act/360) |
| Mid/long | 1Y – 30Y OIS fixed-vs-compounded-SOFR swaps (annual fixed, Act/360) |
| Optional | 30-day Fed Funds futures, 3M SOFR futures (with convexity adjustment) |

### Discount curve

Internally stores `log(DF(t))` at calibration pillar dates.  Three
interpolation modes are available:

| Mode | Description | Forward rate behaviour |
|------|-------------|----------------------|
| `log_linear` | Linear on log(DF) | Piecewise-constant forwards |
| `log_cubic`  | Natural cubic spline on log(DF) | Smooth continuous forwards |
| `linear_zero` | Linear on continuously-compounded zero rates | Monotone but kinked |

`log_linear` is the industry default for bootstrapping (used internally in
both calibration passes); `log_cubic` produces smoother forward curves and
is applied as a post-processing step if requested.

### Two-pass bootstrap

**Pass 1 — Sequential analytic calibration**

Each instrument is calibrated in maturity order, all prior pillars held fixed.

- *Deposits*: `DF(T) = DF(T₀) / (1 + r · dcf)` (exact, O(1))
- *Futures*: same formula using implied forward rate (with convexity adjustment)
- *OIS swaps*: isolate `DF(Tₙ)` from the par condition

  ```
  DF(Tₙ) = [DF(T₀) - S · Σᵢ₌₁ⁿ⁻¹ dcfᵢ · DF(Tᵢ)] / [1 + S · dcfₙ]
  ```

  where `T₀` is the swap effective date and the sum runs over all coupon
  dates already in the curve.  Derivation: the OIS float leg telescopes to
  `DF(T₀) − DF(Tₙ)`; setting `float = fixed` and solving for `DF(Tₙ)`.

**Pass 2 — Global Newton-Raphson refinement**

When calibration instruments have tenor gaps (e.g. 5Y, 7Y, 10Y but no 6Y),
the 6Y coupon of the 7Y swap is interpolated from the log-linear curve.
After adding the 7Y pillar, re-interpolation shifts that intermediate DF,
breaking the 7Y residual.

The global solver treats `x = [log DF(T₁), …, log DF(Tₘ)]` (OIS pillars
only) as unknowns and solves `r(x) = 0` where `rᵢ = par_rate(swap_i) −
fixed_rate_i`.  The Jacobian is approximated by central finite differences.
A simple line-search damps the step if the residual norm increases.

Convergence: `‖r‖∞ < 10⁻⁶ bps` on the full 20-instrument set.

### OIS swap valuation

```
PV_float = DF(T_eff) − DF(T_mat)   [telescoping sum of period float NPVs]

PV_fixed = S · Σᵢ dcfᵢ · DF(Tᵢ)   [annual Act/360 coupons]

NPV      = PV_float − PV_fixed

Par rate = PV_float / annuity
```

### Coupon schedules

Generated backward from maturity in steps of `freq_months`, with
end-of-month anchoring via `ModifiedFollowing` adjustment.  Short front
stubs (period shorter than `freq_months − 1` months between effective date
and the first regular coupon) are absorbed into the first period, consistent
with ISDA 2006 definitions Section 4.14(b).

---

## Results

**20-instrument SOFR OIS run — 2025-06-18 (mid-market snapshot)**

```
════════════════════════════════════════════════════════════════════
  OIS CURVE CALIBRATION REPORT
  Reference date : 2025-06-18  |  Interpolation: log_linear
  Status         : ✓ CONVERGED  |  Max residual: 0.0000 bps
════════════════════════════════════════════════════════════════════
  Label         Maturity     Market     Model      Residual      DF
  ──────────────────────────────────────────────────────────────────
  SOFR_ON      2025-06-20   533.00b   533.00b     -0.0000b  0.99970398
  SOFR_1W      2025-06-27   532.50b   532.50b     -0.0000b  0.99866994
  ...
  OIS_5Y       2030-06-20   434.00b   434.00b     +0.0000b  0.80677780
  OIS_10Y      2035-06-20   420.00b   420.00b     -0.0000b  0.66029659
  OIS_30Y      2055-06-21   414.00b   414.00b     -0.0000b  0.29349163
```

**5Y payer OIS, N=10mm — Bucketed DV01**

```
  Bucket        DV01 ($)
  ──────────────────────
  OIS_5Y        +4,444.87   ████████████████████
  All others     ~0.00
  ──────────────────────
  Total DV01    +4,444.87
  Parallel DV01 +4,444.05
```

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

### Library usage

```python
from ois import Calendar, DayCount, Deposit, OISSwap, OISBootstrapper

cal = Calendar("US")
ref = date(2025, 6, 18)
T2  = cal.advance(ref, n_days=2)

instruments = [
    Deposit("ON",   ref, cal.advance(ref, n_days=1),   rate=0.0533),
    Deposit("3M",   T2,  cal.advance(T2,  n_months=3), rate=0.0527),
    OISSwap("1Y",   T2,  cal.advance(T2,  n_months=12),  fixed_rate=0.0498, calendar=cal),
    OISSwap("5Y",   T2,  cal.advance(T2,  n_months=60),  fixed_rate=0.0434, calendar=cal),
    OISSwap("10Y",  T2,  cal.advance(T2,  n_months=120), fixed_rate=0.0420, calendar=cal),
]

result = OISBootstrapper(ref, instruments, cal).run()
curve  = result.curve

curve.discount(date(2030, 6, 20))           # → 0.8068
curve.zero_rate(date(2030, 6, 20))          # → 0.04288 (continuous)
curve.forward_rate(date(2030, 6, 20),
                   date(2030, 9, 20))       # → 0.0393 (simple, Act/360)
```

---

## References

1. Ametrano & Bianchetti (2013). *Everything You Always Wanted to Know About Multiple Interest Rate Curve Bootstrapping But Were Afraid to Ask.*
2. Brigo & Mercurio (2006). *Interest Rate Models — Theory and Practice*, Ch. 1–2.
3. Hagan & West (2006). *Interpolation Methods for Curve Construction*. Applied Mathematical Finance.
4. ISDA (2006). *2006 ISDA Definitions*, Section 4 (Calculation of Fixed Amounts).

