"""Black-Scholes-Merton prices and Greeks against textbook values and identities."""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import ndtr

from quantkit.exceptions import InvalidParameterError
from quantkit.pricing import OptionKind, OptionSpec, bsm_greeks, bsm_price
from quantkit.pricing.black_scholes import norm_cdf, norm_pdf

HULL_CALL = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")


def option_specs(kind: OptionKind) -> st.SearchStrategy[OptionSpec]:
    """Random but well-posed option specifications."""
    return st.builds(
        OptionSpec,
        spot=st.floats(1.0, 500.0),
        strike=st.floats(1.0, 500.0),
        rate=st.floats(-0.05, 0.20),
        vol=st.floats(0.01, 2.0),
        maturity=st.floats(0.01, 10.0),
        dividend_yield=st.floats(0.0, 0.10),
        kind=st.just(kind),
    )


def test_hull_textbook_example() -> None:
    # Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=10%,
    # sigma=20%, T=0.5 gives c = 4.76 and p = 0.81 (quoted to two decimals).
    assert bsm_price(HULL_CALL) == pytest.approx(4.76, abs=0.005)
    assert bsm_price(replace(HULL_CALL, kind="put")) == pytest.approx(0.81, abs=0.005)


def test_hull_greeks_example() -> None:
    # Hull's Greek-letters example: S=49, K=50, r=5%, sigma=20%, T=20 weeks.
    # Reported: price 2.40, delta 0.522, gamma 0.066, vega 12.1, theta -4.31 per
    # year, rho 8.91.
    option = OptionSpec(spot=49, strike=50, rate=0.05, vol=0.20, maturity=20 / 52, kind="call")
    greeks = bsm_greeks(option)
    assert bsm_price(option) == pytest.approx(2.40, abs=0.005)
    assert greeks.delta == pytest.approx(0.522, abs=0.0005)
    assert greeks.gamma == pytest.approx(0.066, abs=0.0005)
    assert greeks.vega == pytest.approx(12.1, abs=0.05)
    assert greeks.theta == pytest.approx(-4.31, abs=0.005)
    assert greeks.rho == pytest.approx(8.91, abs=0.005)


@given(call=option_specs("call"))
def test_put_call_parity(call: OptionSpec) -> None:
    # c - p = S e^(-qT) - K e^(-rT) holds for any model-free European pair.
    put = replace(call, kind="put")
    forward_value = call.spot * math.exp(-call.dividend_yield * call.maturity)
    pv_strike = call.strike * math.exp(-call.rate * call.maturity)
    scale = max(call.spot, call.strike)
    assert bsm_price(call) - bsm_price(put) == pytest.approx(
        forward_value - pv_strike, abs=1e-10 * scale
    )


@given(call=option_specs("call"))
def test_prices_respect_no_arbitrage_bounds(call: OptionSpec) -> None:
    put = replace(call, kind="put")
    pv_spot = call.spot * math.exp(-call.dividend_yield * call.maturity)
    pv_strike = call.strike * math.exp(-call.rate * call.maturity)
    tol = 1e-10 * max(call.spot, call.strike)
    assert max(pv_spot - pv_strike, 0.0) - tol <= bsm_price(call) <= pv_spot + tol
    assert max(pv_strike - pv_spot, 0.0) - tol <= bsm_price(put) <= pv_strike + tol


GREEK_CASES = [
    HULL_CALL,
    replace(HULL_CALL, kind="put"),
    OptionSpec(spot=100, strike=100, rate=0.03, vol=0.25, maturity=1.0, kind="call"),
    OptionSpec(spot=100, strike=120, rate=0.01, vol=0.40, maturity=2.0, kind="call"),
    OptionSpec(
        spot=100, strike=80, rate=0.05, vol=0.30, maturity=0.25, dividend_yield=0.03, kind="put"
    ),
    OptionSpec(
        spot=50, strike=55, rate=-0.01, vol=0.15, maturity=3.0, dividend_yield=0.02, kind="call"
    ),
    OptionSpec(
        spot=1500, strike=1400, rate=0.04, vol=0.20, maturity=0.5, dividend_yield=0.06, kind="put"
    ),
]


def central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("option", GREEK_CASES)
def test_greeks_match_finite_differences(option: OptionSpec) -> None:
    greeks = bsm_greeks(option)

    def price_at_spot(spot: float) -> float:
        return bsm_price(replace(option, spot=spot))

    # Central differences have O(h^2) truncation error and O(eps / h) round-off
    # (O(eps / h^2) for the second derivative), which sets the step sizes.
    delta = central_difference(price_at_spot, option.spot, 1e-5 * option.spot)
    h_gamma = 1e-3 * option.spot
    gamma = (
        price_at_spot(option.spot + h_gamma)
        - 2.0 * bsm_price(option)
        + price_at_spot(option.spot - h_gamma)
    ) / h_gamma**2
    vega = central_difference(lambda v: bsm_price(replace(option, vol=v)), option.vol, 1e-5)
    # Theta is the derivative with respect to calendar time, i.e. minus d/dT.
    theta = -central_difference(
        lambda t: bsm_price(replace(option, maturity=t)), option.maturity, 1e-5
    )
    rho = central_difference(lambda r: bsm_price(replace(option, rate=r)), option.rate, 1e-5)

    assert greeks.delta == pytest.approx(delta, rel=1e-6, abs=1e-9)
    assert greeks.gamma == pytest.approx(gamma, rel=1e-4, abs=1e-9)
    assert greeks.vega == pytest.approx(vega, rel=1e-6, abs=1e-9)
    assert greeks.theta == pytest.approx(theta, rel=1e-6, abs=1e-9)
    assert greeks.rho == pytest.approx(rho, rel=1e-6, abs=1e-9)


@given(call=option_specs("call"))
def test_call_and_put_greek_relations(call: OptionSpec) -> None:
    # Differentiating put-call parity: delta_c - delta_p = e^(-qT), gamma and vega
    # are shared, and rho_c - rho_p = K T e^(-rT).
    put = replace(call, kind="put")
    gc, gp = bsm_greeks(call), bsm_greeks(put)
    assert gc.delta - gp.delta == pytest.approx(math.exp(-call.dividend_yield * call.maturity))
    assert gc.gamma == pytest.approx(gp.gamma)
    assert gc.vega == pytest.approx(gp.vega)
    expected_rho_gap = call.strike * call.maturity * math.exp(-call.rate * call.maturity)
    assert gc.rho - gp.rho == pytest.approx(expected_rho_gap, rel=1e-9, abs=1e-9)
    assert gc.vega >= 0.0
    assert gc.gamma >= 0.0


def test_norm_cdf_matches_scipy_including_far_tails() -> None:
    xs = np.linspace(-37.0, 8.0, 2001)
    ours = np.array([norm_cdf(float(x)) for x in xs])
    np.testing.assert_allclose(ours, ndtr(xs), rtol=1e-12, atol=0.0)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))


def test_deep_out_of_the_money_price_is_tiny_but_positive() -> None:
    option = OptionSpec(spot=100, strike=300, rate=0.01, vol=0.10, maturity=0.5, kind="call")
    assert 0.0 < bsm_price(option) < 1e-20


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", 0.0),
        ("spot", -1.0),
        ("strike", 0.0),
        ("vol", 0.0),
        ("vol", math.nan),
        ("maturity", 0.0),
        ("maturity", math.inf),
        ("rate", math.nan),
        ("dividend_yield", math.inf),
    ],
)
def test_option_spec_rejects_invalid_inputs(field: str, value: float) -> None:
    params: dict[str, Any] = {
        "spot": 42.0,
        "strike": 40.0,
        "rate": 0.10,
        "vol": 0.20,
        "maturity": 0.5,
        "dividend_yield": 0.0,
        "kind": "call",
    }
    params[field] = value
    with pytest.raises(InvalidParameterError, match=field):
        OptionSpec(**params)


def test_option_spec_rejects_unknown_kind() -> None:
    # cast() smuggles a value the type checker would reject, to test the runtime check.
    straddle = cast("OptionKind", "straddle")
    with pytest.raises(InvalidParameterError, match="kind"):
        OptionSpec(spot=1, strike=1, rate=0, vol=0.1, maturity=1, kind=straddle)


def test_option_spec_normalizes_numbers_to_float() -> None:
    option = OptionSpec(spot=np.float64(42), strike=40, rate=0, vol=0.2, maturity=1, kind="put")
    assert type(option.spot) is float
    assert type(option.strike) is float
    np.testing.assert_array_equal(option.payoff(np.array([30.0, 40.0, 50.0])), [10.0, 0.0, 0.0])
