"""CRR binomial trees: convergence to BSM, early exercise, and textbook values."""

import math
from dataclasses import replace
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantkit.exceptions import InvalidParameterError
from quantkit.pricing import ExerciseStyle, OptionSpec, binomial_price, bsm_price

HULL_CALL = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")

CONVERGENCE_CASES = [
    HULL_CALL,
    replace(HULL_CALL, kind="put"),
    OptionSpec(spot=100, strike=110, rate=0.03, vol=0.30, maturity=1.0, kind="call"),
    OptionSpec(
        spot=100, strike=90, rate=0.02, vol=0.25, maturity=2.0, dividend_yield=0.04, kind="put"
    ),
]


def worst_error(option: OptionSpec, first_steps: int, count: int = 10) -> float:
    """Largest |tree - BSM| over ``count`` consecutive step counts."""
    exact = bsm_price(option)
    steps_range = range(first_steps, first_steps + count)
    return max(abs(binomial_price(option, steps=n) - exact) for n in steps_range)


@pytest.mark.parametrize("option", CONVERGENCE_CASES)
def test_european_tree_converges_to_black_scholes(option: OptionSpec) -> None:
    # The CRR error is O(1/steps) but oscillates as tree nodes move relative to
    # the strike, so compare the worst error over windows of consecutive steps.
    coarse = worst_error(option, 50)
    fine = worst_error(option, 1600)
    assert fine * 1600 < 0.05 * option.spot  # error <= C / steps with a modest C
    assert coarse / fine > 10.0  # 32x more steps shrinks the error by over 10x


def test_one_step_tree_matches_hand_calculation() -> None:
    option = replace(HULL_CALL, kind="put")
    u = math.exp(0.20 * math.sqrt(0.5))
    d = 1.0 / u
    p = (math.exp(0.10 * 0.5) - d) / (u - d)
    put_up = max(40 - 42 * u, 0.0)
    put_down = max(40 - 42 * d, 0.0)
    european = math.exp(-0.10 * 0.5) * (p * put_up + (1 - p) * put_down)
    assert binomial_price(option, steps=1) == pytest.approx(european, rel=1e-14)
    # At the root, exercising immediately is worth max(40 - 42, 0) = 0, so the
    # one-step American value is the same.
    assert binomial_price(option, steps=1, exercise="american") == pytest.approx(european)


@pytest.mark.parametrize(
    ("steps", "expected", "decimals"),
    [(5, 4.49, 2), (30, 4.263, 3), (50, 4.272, 3), (100, 4.278, 3), (500, 4.283, 3)],
)
def test_hull_american_put_example(steps: int, expected: float, decimals: int) -> None:
    # Hull's American put on a non-dividend-paying stock: S=K=50, r=10%,
    # sigma=40%, T=5 months. Hull reports 4.49 for a 5-step tree and 4.263,
    # 4.272, 4.278 and 4.283 for 30, 50, 100 and 500 steps.
    option = OptionSpec(spot=50, strike=50, rate=0.10, vol=0.40, maturity=5 / 12, kind="put")
    value = binomial_price(option, steps=steps, exercise="american")
    assert round(value, decimals) == expected


american_cases = st.builds(
    OptionSpec,
    spot=st.floats(20.0, 200.0),
    strike=st.floats(20.0, 200.0),
    rate=st.floats(0.0, 0.15),
    vol=st.floats(0.05, 1.0),
    maturity=st.floats(0.05, 3.0),
    dividend_yield=st.floats(0.0, 0.10),
    kind=st.just("put"),
)


@given(put=american_cases)
def test_american_put_is_worth_at_least_european_and_intrinsic(put: OptionSpec) -> None:
    american = binomial_price(put, steps=200, exercise="american")
    european = binomial_price(put, steps=200, exercise="european")
    assert american >= european - 1e-12
    assert american >= put.strike - put.spot - 1e-12


def test_american_put_carries_an_early_exercise_premium() -> None:
    put = OptionSpec(spot=50, strike=50, rate=0.10, vol=0.40, maturity=5 / 12, kind="put")
    premium = binomial_price(put, steps=1000, exercise="american") - bsm_price(put)
    assert premium > 0.2


@given(call=american_cases.map(lambda o: replace(o, kind="call", dividend_yield=0.0)))
def test_american_call_equals_european_without_dividends(call: OptionSpec) -> None:
    # With q = 0 and r >= 0 a call is worth more alive than exercised (Merton 1973),
    # so the American and European trees must agree.
    american = binomial_price(call, steps=200, exercise="american")
    european = binomial_price(call, steps=200, exercise="european")
    assert american == pytest.approx(european, rel=1e-12, abs=1e-12)


def test_american_call_exceeds_european_with_high_dividends() -> None:
    call = OptionSpec(
        spot=100, strike=80, rate=0.02, vol=0.20, maturity=1.0, dividend_yield=0.10, kind="call"
    )
    american = binomial_price(call, steps=500, exercise="american")
    assert american > binomial_price(call, steps=500) + 0.1


@pytest.mark.parametrize("steps", [0, -5])
def test_rejects_invalid_step_counts(steps: int) -> None:
    with pytest.raises(InvalidParameterError, match="steps"):
        binomial_price(HULL_CALL, steps=steps)


def test_rejects_non_integer_steps() -> None:
    with pytest.raises(InvalidParameterError, match="steps"):
        binomial_price(HULL_CALL, steps=cast("int", 10.5))


def test_rejects_unknown_exercise_style() -> None:
    with pytest.raises(InvalidParameterError, match="exercise"):
        binomial_price(HULL_CALL, exercise=cast("ExerciseStyle", "bermudan"))


def test_rejects_trees_with_invalid_probabilities() -> None:
    # A large drift over one coarse step pushes the up-probability above one.
    option = OptionSpec(spot=100, strike=100, rate=0.5, vol=0.01, maturity=1.0, kind="call")
    with pytest.raises(InvalidParameterError, match="increase steps"):
        binomial_price(option, steps=1)
