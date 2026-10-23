"""Implied volatility: round trips through BSM and no-arbitrage error handling."""

import math
from dataclasses import replace
from typing import cast

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from quantkit.exceptions import ArbitrageBoundError, ConvergenceError, InvalidParameterError
from quantkit.pricing import (
    OptionKind,
    OptionSpec,
    bsm_greeks,
    bsm_price,
    implied_vol,
    no_arbitrage_bounds,
)


def solve_iv(price: float, option: OptionSpec) -> float:
    """Implied vol for ``price`` using every input of ``option`` except its vol."""
    return implied_vol(
        price,
        spot=option.spot,
        strike=option.strike,
        rate=option.rate,
        maturity=option.maturity,
        kind=option.kind,
        dividend_yield=option.dividend_yield,
    )


@given(
    option=st.builds(
        OptionSpec,
        spot=st.just(100.0),
        strike=st.floats(50.0, 200.0),
        rate=st.floats(-0.02, 0.15),
        vol=st.floats(0.02, 3.0),
        maturity=st.floats(0.05, 5.0),
        dividend_yield=st.floats(0.0, 0.08),
        kind=st.sampled_from(["call", "put"]),
    )
)
def test_round_trip_recovers_volatility(option: OptionSpec) -> None:
    # When vega is essentially zero the price carries no information about
    # volatility (it is pinned to a no-arbitrage bound), so skip those cases.
    assume(bsm_greeks(option).vega > 1e-4 * option.spot)
    price = bsm_price(option)
    iv = solve_iv(price, option)
    assert iv == pytest.approx(option.vol, rel=1e-8)
    assert bsm_price(replace(option, vol=iv)) == pytest.approx(price, rel=1e-10)


def test_hull_example_round_trip() -> None:
    option = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")
    assert solve_iv(bsm_price(option), option) == pytest.approx(0.20, rel=1e-12)
    # The textbook price rounded to cents still implies a volatility of ~20%.
    assert solve_iv(4.76, option) == pytest.approx(0.20, abs=1e-3)


def test_extreme_but_valid_volatility_round_trips() -> None:
    option = OptionSpec(spot=100, strike=100, rate=0.02, vol=6.0, maturity=1.0, kind="call")
    assert solve_iv(bsm_price(option), option) == pytest.approx(6.0, rel=1e-8)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_prices_outside_bounds_raise_typed_error(kind: OptionKind) -> None:
    option = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind=kind)
    lower, upper = no_arbitrage_bounds(spot=42, strike=40, rate=0.10, maturity=0.5, kind=kind)
    for bad_price in (upper + 0.01, upper, lower, lower - 0.01, -1.0):
        with pytest.raises(ArbitrageBoundError) as info:
            solve_iv(bad_price, option)
        assert info.value.lower == lower
        assert info.value.upper == upper
        assert "no-arbitrage" in str(info.value)


def test_arbitrage_error_is_a_value_error() -> None:
    option = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")
    with pytest.raises(ValueError, match="no-arbitrage"):
        solve_iv(50.0, option)


def test_bounds_match_textbook_formulas() -> None:
    lower, upper = no_arbitrage_bounds(
        spot=42, strike=40, rate=0.10, maturity=0.5, kind="call", dividend_yield=0.02
    )
    pv_spot = 42 * math.exp(-0.02 * 0.5)
    pv_strike = 40 * math.exp(-0.10 * 0.5)
    assert lower == pytest.approx(pv_spot - pv_strike)
    assert upper == pytest.approx(pv_spot)
    lower, upper = no_arbitrage_bounds(
        spot=42, strike=40, rate=0.10, maturity=0.5, kind="put", dividend_yield=0.02
    )
    assert lower == 0.0
    assert upper == pytest.approx(pv_strike)


def test_volatility_above_search_interval_raises_convergence_error() -> None:
    option = OptionSpec(spot=100, strike=100, rate=0.0, vol=15.0, maturity=1.0, kind="call")
    with pytest.raises(ConvergenceError, match="exceeds"):
        solve_iv(bsm_price(option), option)


def test_volatility_below_search_interval_raises_convergence_error() -> None:
    # At-the-money-forward option with an absurdly small volatility: its price is
    # inside the bounds but indistinguishable from intrinsic value at 1e-6 vol.
    option = OptionSpec(spot=40, strike=40, rate=0.0, vol=1e-9, maturity=1.0, kind="call")
    with pytest.raises(ConvergenceError, match="below"):
        solve_iv(bsm_price(option), option)


def test_invalid_inputs_are_rejected() -> None:
    option = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")
    with pytest.raises(InvalidParameterError, match="price"):
        solve_iv(math.nan, option)
    with pytest.raises(InvalidParameterError, match="kind"):
        no_arbitrage_bounds(
            spot=42, strike=40, rate=0.1, maturity=0.5, kind=cast("OptionKind", "swap")
        )
    with pytest.raises(InvalidParameterError, match="maturity"):
        no_arbitrage_bounds(spot=42, strike=40, rate=0.1, maturity=-1.0, kind="call")
