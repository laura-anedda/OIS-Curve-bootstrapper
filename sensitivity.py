from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Union

from .bootstrap import OISBootstrapper, Instrument
from .conventions import Calendar
from .curve import DiscountCurve, InterpolationMethod
from .instruments import OISSwap, Deposit, FedFundsFuture, SOFRFuture


@dataclass
class BucketDV01:
    """
    DV01 of a portfolio position against each calibration instrument.

    dv01[label] = change in position PV for +1 bp shift of instrument `label`,
                  all other instruments held fixed.
    """
    position_label: str
    dv01: Dict[str, float]          # {instrument_label: dv01_ccy}
    total_dv01: float


@dataclass
class CurveSensitivity:
    """Full DV01 grid across all instruments for a set of positions."""
    reference_date: date
    bucket_dv01s: List[BucketDV01]


def bump_and_reprice(
    reference_date: date,
    instruments: List[Instrument],
    calendar: Calendar,
    interpolation: str,
    bump_label: str,
    bump_bps: float = 1.0,
) -> DiscountCurve:
    """
    Return a bumped curve where instrument `bump_label` is shifted by `bump_bps`.
    All other instruments are unchanged.
    """
    bumped_instruments = []
    for inst in instruments:
        inst_copy = copy.deepcopy(inst)
        if inst_copy.label == bump_label:
            if isinstance(inst_copy, Deposit):
                inst_copy.rate += bump_bps / 1e4
            elif isinstance(inst_copy, (FedFundsFuture, SOFRFuture)):
                inst_copy.price -= bump_bps / 100.0  # price down = rate up
            elif isinstance(inst_copy, OISSwap):
                inst_copy.fixed_rate += bump_bps / 1e4
        bumped_instruments.append(inst_copy)

    result = OISBootstrapper(
        reference_date, bumped_instruments, calendar, interpolation
    ).run()
    return result.curve


def compute_dv01(
    reference_date: date,
    base_curve: DiscountCurve,
    instruments: List[Instrument],
    calendar: Calendar,
    interpolation: str,
    swap: OISSwap,
    notional: float = 1_000_000.0,
    bump_bps: float = 1.0,
) -> BucketDV01:
    """
    Compute bucketed DV01 of a single swap position via bump-and-reprice.

    For each instrument in the calibration set, bump its rate by `bump_bps`,
    re-bootstrap the curve, and compute the change in swap NPV.
    """
    base_npv = swap.npv(base_curve) * notional
    dv01: Dict[str, float] = {}

    for inst in instruments:
        bumped_curve = bump_and_reprice(
            reference_date, instruments, calendar, interpolation,
            bump_label=inst.label, bump_bps=bump_bps,
        )
        bumped_npv  = swap.npv(bumped_curve) * notional
        dv01[inst.label] = (bumped_npv - base_npv) / bump_bps  # per 1 bp

    total = sum(dv01.values())
    return BucketDV01(position_label=swap.label, dv01=dv01, total_dv01=total)


def parallel_shift_dv01(
    reference_date: date,
    base_curve: DiscountCurve,
    instruments: List[Instrument],
    calendar: Calendar,
    interpolation: str,
    swap: OISSwap,
    notional: float = 1_000_000.0,
    bump_bps: float = 1.0,
) -> float:
    """Parallel-shift DV01: bump all instruments simultaneously."""
    base_npv = swap.npv(base_curve) * notional

    bumped_instruments = copy.deepcopy(instruments)
    for inst in bumped_instruments:
        if isinstance(inst, Deposit):
            inst.rate += bump_bps / 1e4
        elif isinstance(inst, (FedFundsFuture, SOFRFuture)):
            inst.price -= bump_bps / 100.0
        elif isinstance(inst, OISSwap):
            inst.fixed_rate += bump_bps / 1e4

    bumped_curve = OISBootstrapper(
        reference_date, bumped_instruments, calendar, interpolation
    ).run().curve

    bumped_npv = swap.npv(bumped_curve) * notional
    return (bumped_npv - base_npv) / bump_bps
