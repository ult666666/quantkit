"""Monte Carlo pricing: agreement with BSM, variance reduction, reproducibility."""

from dataclasses import replace

import numpy as np
import pytest

from quantkit.exceptions import InvalidParameterError
from quantkit.pricing import OptionSpec, bsm_price, monte_carlo_price
from quantkit.pricing.monte_carlo import Z_95

CASES = [
    OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call"),
    OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="put"),
    OptionSpec(
        spot=100, strike=120, rate=0.03, vol=0.35, maturity=2.0, dividend_yield=0.02, kind="call"
    ),
    OptionSpec(spot=100, strike=95, rate=-0.01, vol=0.15, maturity=0.25, kind="put"),
]


@pytest.mark.parametrize("option", CASES)
@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_estimate_is_within_three_standard_errors_of_bsm(
    option: OptionSpec, antithetic: bool, seed: int
) -> None:
    result = monte_carlo_price(option, n_paths=100_000, antithetic=antithetic, seed=seed)
    assert abs(result.price - bsm_price(option)) < 3.0 * result.std_error


def test_confidence_intervals_have_nominal_coverage() -> None:
    # If the standard error is right, about 95% of independent 95% intervals
    # contain the true (BSM) price. With 400 trials the binomial standard
    # deviation of the hit rate is ~1.1%, so [92%, 98%] is a ~3-sigma band.
    option = CASES[0]
    exact = bsm_price(option)
    hits = 0
    for seed in range(400):
        result = monte_carlo_price(option, n_paths=2_000, seed=seed)
        if result.ci_low <= exact <= result.ci_high:
            hits += 1
    assert 0.92 <= hits / 400 <= 0.98


@pytest.mark.parametrize("option", CASES)
def test_antithetic_variates_reduce_the_standard_error(option: OptionSpec) -> None:
    # Same number of payoff evaluations for both estimators.
    plain = monte_carlo_price(option, n_paths=50_000, antithetic=False, seed=7)
    paired = monte_carlo_price(option, n_paths=50_000, antithetic=True, seed=7)
    assert paired.std_error < plain.std_error


def test_result_fields_are_consistent() -> None:
    result = monte_carlo_price(CASES[0], n_paths=10_000, seed=0)
    assert result.n_paths == 10_000
    assert result.ci_low < result.price < result.ci_high
    assert result.ci_high - result.ci_low == pytest.approx(2 * Z_95 * result.std_error)


def test_seed_makes_results_reproducible() -> None:
    first = monte_carlo_price(CASES[0], n_paths=10_000, seed=123)
    second = monte_carlo_price(CASES[0], n_paths=10_000, seed=123)
    other = monte_carlo_price(CASES[0], n_paths=10_000, seed=124)
    assert first == second
    assert first.price != other.price


def test_accepts_a_numpy_generator() -> None:
    from_seed = monte_carlo_price(CASES[0], n_paths=10_000, seed=5)
    from_generator = monte_carlo_price(CASES[0], n_paths=10_000, seed=np.random.default_rng(5))
    assert from_seed == from_generator


def test_zero_payoff_everywhere_gives_zero_price_and_error() -> None:
    # Strike so far out of the money that no simulated path finishes in the money.
    option = replace(CASES[0], strike=10_000.0)
    result = monte_carlo_price(option, n_paths=1_000, seed=0)
    assert result.price == 0.0
    assert result.std_error == 0.0


def test_rejects_odd_path_counts_with_antithetic_sampling() -> None:
    with pytest.raises(InvalidParameterError, match="even"):
        monte_carlo_price(CASES[0], n_paths=1_001, antithetic=True)
    # Without antithetic variates an odd count is fine.
    assert monte_carlo_price(CASES[0], n_paths=1_001, antithetic=False, seed=0).n_paths == 1_001


def test_rejects_too_few_paths() -> None:
    with pytest.raises(InvalidParameterError, match="n_paths"):
        monte_carlo_price(CASES[0], n_paths=2)
