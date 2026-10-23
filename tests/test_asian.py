"""Asian options: geometric closed form and the control-variate Monte Carlo."""

import math
from dataclasses import replace

import numpy as np
import pytest

from quantkit.exceptions import InvalidParameterError
from quantkit.pricing import (
    OptionSpec,
    asian_monte_carlo_price,
    bsm_price,
    geometric_asian_price,
)

CALL = OptionSpec(spot=100, strike=100, rate=0.05, vol=0.20, maturity=1.0, kind="call")
PUT = replace(CALL, kind="put")
CASES = [CALL, PUT, replace(CALL, strike=110, dividend_yield=0.02, vol=0.35)]


@pytest.mark.parametrize("option", CASES)
def test_one_fixing_at_expiry_reduces_to_black_scholes(option: OptionSpec) -> None:
    assert geometric_asian_price(option, n_fixings=1) == pytest.approx(
        bsm_price(option), rel=1e-12
    )


@pytest.mark.parametrize("option", CASES)
@pytest.mark.parametrize("n_fixings", [4, 12, 52])
def test_geometric_closed_form_matches_independent_simulation(
    option: OptionSpec, n_fixings: int
) -> None:
    # Simulate the geometric average directly (without the library's simulator)
    # and check the closed form lies within 3 standard errors.
    rng = np.random.default_rng(n_fixings)
    dt = option.maturity / n_fixings
    drift = (option.rate - option.dividend_yield - 0.5 * option.vol**2) * dt
    shocks = option.vol * math.sqrt(dt) * rng.standard_normal((200_000, n_fixings))
    geometric_mean = option.spot * np.exp(np.cumsum(drift + shocks, axis=1).mean(axis=1))
    payoffs = math.exp(-option.rate * option.maturity) * option.payoff(geometric_mean)
    std_error = payoffs.std(ddof=1) / math.sqrt(payoffs.size)
    exact = geometric_asian_price(option, n_fixings=n_fixings)
    assert abs(payoffs.mean() - exact) < 3 * std_error


@pytest.mark.parametrize("option", CASES)
def test_control_variate_agrees_with_plain_monte_carlo_and_is_far_more_precise(
    option: OptionSpec,
) -> None:
    controlled = asian_monte_carlo_price(option, n_fixings=12, n_paths=200_000, seed=1)
    plain = asian_monte_carlo_price(
        option, n_fixings=12, n_paths=200_000, seed=2, control_variate=False
    )
    combined_error = math.hypot(controlled.std_error, plain.std_error)
    assert abs(controlled.price - plain.price) < 3 * combined_error
    assert controlled.std_error < plain.std_error / 10


def test_arithmetic_average_bounds() -> None:
    # AM >= GM pathwise, so an arithmetic call is worth at least the geometric
    # one (and a put at most). Averaging also damps volatility, so with r >= q
    # the Asian call is cheaper than the European call.
    call = asian_monte_carlo_price(CALL, n_fixings=12, seed=3)
    put = asian_monte_carlo_price(PUT, n_fixings=12, seed=3)
    assert call.price > geometric_asian_price(CALL, n_fixings=12)
    assert put.price < geometric_asian_price(PUT, n_fixings=12)
    assert call.price < bsm_price(CALL)


def test_seeded_results_are_reproducible() -> None:
    first = asian_monte_carlo_price(CALL, n_fixings=12, n_paths=1_000, seed=9)
    assert first == asian_monte_carlo_price(CALL, n_fixings=12, n_paths=1_000, seed=9)


def test_deep_out_of_the_money_option_with_no_payoffs() -> None:
    result = asian_monte_carlo_price(replace(CALL, strike=1_000.0), n_fixings=4, n_paths=1_000)
    assert result.price == pytest.approx(0.0, abs=1e-12)


def test_rejects_invalid_fixings() -> None:
    with pytest.raises(InvalidParameterError, match="n_fixings"):
        geometric_asian_price(CALL, n_fixings=0)
    with pytest.raises(InvalidParameterError, match="n_fixings"):
        asian_monte_carlo_price(CALL, n_fixings=0)
