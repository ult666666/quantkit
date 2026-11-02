"""Exception backtests against hand calculations, SciPy and brute-force enumeration."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binom, chi2

from quantkit.exceptions import InvalidParameterError
from quantkit.var import (
    BASEL_DAYS,
    TransitionCounts,
    backtest_var,
    basel_traffic_light,
    christoffersen_test,
    conditional_coverage_test,
    kupiec_test,
    null_rejection_rates,
    transition_counts,
    var_exceptions,
)

hit_sequences = st.lists(st.booleans(), min_size=2, max_size=400)
levels = st.floats(0.8, 0.999)


def hits(n_days: int, exception_days: list[int]) -> np.ndarray:
    flags = np.zeros(n_days, dtype=bool)
    flags[exception_days] = True
    return flags


# ------------------------------------------------------------- Kupiec (1995)


def test_kupiec_by_hand() -> None:
    # Six exceptions in 250 days at 99% (2.5 expected), Kupiec's LR_uc:
    # 2 [6 ln(6 / 2.5) + 244 ln(244 / 247.5)] = 2 [5.25281 - 3.47514] = 3.5554.
    result = kupiec_test(hits(250, [3, 50, 90, 130, 170, 210]), level=0.99)
    expected = 2 * (6 * math.log(6 / 2.5) + 244 * math.log(244 / 247.5))
    assert result.statistic == pytest.approx(expected, rel=1e-12)
    assert result.statistic == pytest.approx(3.5554, abs=1e-4)
    assert result.dof == 1
    assert result.p_value == pytest.approx(float(chi2.sf(expected, 1)), rel=1e-12)
    # Seven exceptions cross the 5% critical value 3.84: 2 [7 ln 2.8 + 243 ln(243 / 247.5)].
    seven = kupiec_test(hits(250, [3, 50, 90, 130, 170, 210, 240]), level=0.99)
    assert seven.statistic == pytest.approx(
        2 * (7 * math.log(2.8) + 243 * math.log(243 / 247.5)), rel=1e-12
    )
    assert seven.p_value < 0.05 < result.p_value


def test_kupiec_with_no_exceptions_or_only_exceptions() -> None:
    # pi_hat = 0 or 1 makes the alternative likelihood exactly 1 (0^0 = 1).
    none = kupiec_test(np.zeros(250, dtype=bool), level=0.99)
    assert none.statistic == pytest.approx(-2 * 250 * math.log(0.99), rel=1e-12)  # 5.025
    assert none.p_value == pytest.approx(float(chi2.sf(none.statistic, 1)), rel=1e-12)
    assert none.p_value < 0.05  # two-sided: too few exceptions also reject
    only = kupiec_test(np.ones(20, dtype=bool), level=0.99)
    assert only.statistic == pytest.approx(-2 * 20 * math.log(0.01), rel=1e-12)
    assert kupiec_test([True], level=0.5).statistic == pytest.approx(2 * math.log(2))


@given(n_days=st.integers(1, 2000), share=st.floats(0.0, 1.0), level=st.floats(0.5, 0.9999))
def test_kupiec_matches_binomial_log_likelihoods(n_days: int, share: float, level: float) -> None:
    """An independent formula: the binomial coefficient cancels in the ratio."""
    x = round(share * n_days)
    result = kupiec_test(hits(n_days, list(range(x))), level=level)
    expected = 2 * (binom.logpmf(x, n_days, x / n_days) - binom.logpmf(x, n_days, 1 - level))
    assert result.statistic == pytest.approx(float(expected), rel=1e-9, abs=1e-9)


# ------------------------------------------------------- Christoffersen (1998)

CLUSTERED = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]


def test_transition_counts_by_hand() -> None:
    assert transition_counts(CLUSTERED) == TransitionCounts(n00=4, n01=2, n10=2, n11=3)
    assert transition_counts([True, False]) == TransitionCounts(n00=0, n01=0, n10=1, n11=0)


def test_christoffersen_by_hand() -> None:
    # n00 = 4, n01 = 2, n10 = 2, n11 = 3: pi_01 = 2/6, pi_11 = 3/5, pooled pi = 5/11.
    markov = 4 * math.log(2 / 3) + 2 * math.log(1 / 3) + 2 * math.log(2 / 5) + 3 * math.log(3 / 5)
    iid = 6 * math.log(6 / 11) + 5 * math.log(5 / 11)
    independence = christoffersen_test(CLUSTERED)
    assert independence.statistic == pytest.approx(2 * (markov - iid), rel=1e-12)  # 0.7899
    assert independence.statistic == pytest.approx(0.7899, abs=1e-4)
    assert independence.p_value == pytest.approx(float(chi2.sf(2 * (markov - iid), 1)))
    # Conditional coverage at 95%: L(p) = 0.95^6 * 0.05^5 over the 11 transitions.
    coverage = conditional_coverage_test(CLUSTERED, level=0.95)
    expected = 2 * (markov - 6 * math.log(0.95) - 5 * math.log(0.05))
    assert coverage.statistic == pytest.approx(expected, rel=1e-12)  # 16.2046
    assert coverage.dof == 2
    assert coverage.p_value == pytest.approx(float(chi2.sf(expected, 2)), rel=1e-12)


def direct_markov_loglik(flags: list[bool]) -> float:
    """Markov-chain log-likelihood by walking the day pairs (shares no code)."""
    pairs = list(itertools.pairwise(flags))
    total = 0.0
    for state in (False, True):
        following = [after for before, after in pairs if before == state]
        if following:
            pi = sum(following) / len(following)
            total += sum(math.log(pi if after else 1 - pi) for after in following)
    return total


@given(flags=hit_sequences)
def test_independence_matches_a_direct_likelihood(flags: list[bool]) -> None:
    after = flags[1:]
    pi = sum(after) / len(after)
    iid = sum(math.log(pi if hit else 1 - pi) for hit in after)
    expected = max(0.0, 2 * (direct_markov_loglik(flags) - iid))
    assert christoffersen_test(flags).statistic == pytest.approx(expected, abs=1e-9)


@given(flags=hit_sequences, level=levels)
def test_conditional_coverage_is_kupiec_on_days_two_onward_plus_independence(
    flags: list[bool], level: float
) -> None:
    # Christoffersen (1998): LR_cc = LR_uc + LR_ind when LR_uc conditions on the first day.
    combined = conditional_coverage_test(flags, level=level).statistic
    parts = kupiec_test(flags[1:], level=level).statistic + christoffersen_test(flags).statistic
    assert combined == pytest.approx(parts, rel=1e-9, abs=1e-9)


@given(flags=hit_sequences, level=levels)
def test_p_values_match_scipy_chi2(flags: list[bool], level: float) -> None:
    for result in (
        kupiec_test(flags, level=level),
        christoffersen_test(flags),
        conditional_coverage_test(flags, level=level),
    ):
        assert result.statistic >= 0
        assert result.p_value == pytest.approx(
            float(chi2.sf(result.statistic, result.dof)), rel=1e-9, abs=1e-15
        )


def test_edge_cases_have_finite_statistics() -> None:
    quiet, busy = np.zeros(250, dtype=bool), np.ones(250, dtype=bool)
    for flags, p_tail in ((quiet, 0.99), (busy, 0.01)):
        # No transition between states: both likelihoods are equal, LR_ind = 0.
        assert christoffersen_test(flags).statistic == 0.0
        assert christoffersen_test(flags).p_value == 1.0
        # LR_cc then reduces to Kupiec on the 249 transitions.
        coverage = conditional_coverage_test(flags, level=0.99)
        assert coverage.statistic == pytest.approx(-2 * 249 * math.log(p_tail), rel=1e-12)
    # Isolated exceptions only (n11 = 0): pi_11 = 0 fits exactly and LR_ind is finite.
    isolated = hits(250, [10, 100, 200])
    counts = transition_counts(isolated)
    assert counts == TransitionCounts(n00=243, n01=3, n10=3, n11=0)
    markov = 243 * math.log(243 / 246) + 3 * math.log(3 / 246)  # the n10 term is 3 ln 1 = 0
    iid = 246 * math.log(246 / 249) + 3 * math.log(3 / 249)
    assert christoffersen_test(isolated).statistic == pytest.approx(2 * (markov - iid))  # 0.0732
    # The only exception on the last day: no pair starts with an exception at all.
    last = hits(10, [9])
    assert transition_counts(last) == TransitionCounts(n00=8, n01=1, n10=0, n11=0)
    assert christoffersen_test(last).statistic == 0.0
    assert math.isfinite(conditional_coverage_test(last, level=0.9).statistic)


def test_backtest_var_collects_everything() -> None:
    flags = pd.Series(hits(250, [5, 6, 100, 180]), index=pd.bdate_range("2024-01-01", periods=250))
    result = backtest_var(flags, level=0.99)
    assert result.n_observations == 250
    assert result.n_exceptions == 4
    assert result.expected_exceptions == pytest.approx(2.5)
    assert result.exception_rate == pytest.approx(0.016)
    assert result.transitions == TransitionCounts(n00=242, n01=3, n10=3, n11=1)
    assert result.unconditional == kupiec_test(flags, level=0.99)
    assert result.independence == christoffersen_test(flags)
    assert result.conditional == conditional_coverage_test(flags, level=0.99)


def test_var_exceptions() -> None:
    returns = pd.Series([-0.03, -0.02, 0.01], index=pd.bdate_range("2024-01-01", periods=3))
    var = pd.Series(0.02, index=returns.index)
    assert list(var_exceptions(returns, var)) == [True, False, False]  # -0.02 is not < -0.02
    assert var_exceptions(returns, var).index.equals(returns.index)
    assert list(var_exceptions([-0.05, 0.0], [0.02, 0.02])) == [True, False]
    with pytest.raises(InvalidParameterError, match="same length"):
        var_exceptions([0.01, 0.02], [0.02])
    with pytest.raises(InvalidParameterError, match="same index"):
        var_exceptions(returns, var.set_axis(pd.bdate_range("2025-01-01", periods=3)))


def test_rejects_invalid_exception_series() -> None:
    with pytest.raises(InvalidParameterError, match="0/1"):
        kupiec_test([0, 2, 1], level=0.99)
    with pytest.raises(InvalidParameterError, match="at least 2"):
        christoffersen_test([True])
    with pytest.raises(InvalidParameterError, match="one-dimensional"):
        backtest_var(np.zeros((3, 2), dtype=bool), level=0.99)
    with pytest.raises(InvalidParameterError, match="level"):
        kupiec_test([0, 1], level=1.5)
    assert kupiec_test([0, 1, 0], level=0.9) == kupiec_test([False, True, False], level=0.9)


# -------------------------------------------------------- exact null sizes


@pytest.mark.parametrize("n_days", [2, 3, 7, 12])
@pytest.mark.parametrize("level", [0.6, 0.9])
def test_exact_sizes_match_brute_force_enumeration(n_days: int, level: float) -> None:
    """Sum the probability of every one of the 2^T exception sequences that is rejected."""
    p = 1 - level
    rejected = np.zeros(3)
    for flags in itertools.product((False, True), repeat=n_days):
        x = sum(flags)
        probability = p**x * (1 - p) ** (n_days - x)
        tests = (
            kupiec_test(flags, level=level),
            christoffersen_test(flags),
            conditional_coverage_test(flags, level=level),
        )
        rejected += probability * np.array([test.p_value < 0.05 for test in tests])
    rates = null_rejection_rates(n_days, level=level)
    np.testing.assert_allclose(
        [rates.unconditional, rates.independence, rates.conditional], rejected, atol=1e-12
    )


@pytest.mark.parametrize("n_days", [250, 1000])
def test_exact_kupiec_size_matches_the_binomial_distribution(n_days: int) -> None:
    critical = float(chi2.ppf(0.95, 1))
    counts = np.arange(n_days + 1)
    statistics = [
        2 * (binom.logpmf(x, n_days, x / n_days) - binom.logpmf(x, n_days, 0.01)) for x in counts
    ]
    exact = float(binom.pmf(counts, n_days, 0.01)[np.array(statistics) > critical].sum())
    assert null_rejection_rates(n_days, level=0.99).unconditional == pytest.approx(
        exact, abs=1e-12
    )


def test_nominal_five_percent_tests_are_far_from_five_percent_at_250_days() -> None:
    rates = null_rejection_rates(250, level=0.99)
    # Zero exceptions alone (probability 0.99^250 = 8.1%) already rejects LR_uc.
    assert rates.unconditional == pytest.approx(0.0948, abs=5e-5)
    assert rates.unconditional > 0.99**250
    assert rates.independence == pytest.approx(0.0140, abs=5e-5)
    assert rates.conditional == pytest.approx(0.0110, abs=5e-5)


def test_exact_sizes_refuse_huge_enumerations() -> None:
    with pytest.raises(InvalidParameterError, match="too slow"):
        null_rejection_rates(20_000, level=0.9)
    with pytest.raises(InvalidParameterError, match="n_observations"):
        null_rejection_rates(1, level=0.99)


# --------------------------------------------- Basel traffic light (BCBS 1996)

# Table 2 of the Basel Committee's 1996 backtesting framework, 250 observations:
# exceptions -> (zone, plus factor, cumulative probability in percent).
BASEL_TABLE = {
    0: ("green", 0.00, 8.11),
    1: ("green", 0.00, 28.58),
    2: ("green", 0.00, 54.32),
    3: ("green", 0.00, 75.81),
    4: ("green", 0.00, 89.22),
    5: ("yellow", 0.40, 95.88),
    6: ("yellow", 0.50, 98.63),
    7: ("yellow", 0.65, 99.60),
    8: ("yellow", 0.75, 99.89),
    9: ("yellow", 0.85, 99.97),
    10: ("red", 1.00, 99.99),
}


@pytest.mark.parametrize(("count", "row"), BASEL_TABLE.items())
def test_traffic_light_reproduces_the_basel_table(
    count: int, row: tuple[str, float, float]
) -> None:
    zone, plus_factor, cumulative = row
    light = basel_traffic_light(count)
    assert (light.zone, light.n_exceptions, light.plus_factor) == (zone, count, plus_factor)
    assert light.multiplier == pytest.approx(3.0 + plus_factor)
    # The published probabilities are binomial(250, 1%) CDFs rounded to 0.01%.
    assert round(100 * float(binom.cdf(count, 250, 0.01)), 2) == cumulative


def test_zones_follow_the_cumulative_probability_rule() -> None:
    # Green below a 95% cumulative probability, red from 99.99%, yellow between.
    for count in range(BASEL_DAYS + 1):
        cumulative = float(binom.cdf(count, BASEL_DAYS, 0.01))
        expected = "green" if cumulative < 0.95 else "yellow" if cumulative < 0.9999 else "red"
        assert basel_traffic_light(count).zone == expected


def test_traffic_light_rejects_impossible_counts() -> None:
    for bad in (-1, 251):
        with pytest.raises(InvalidParameterError, match="n_exceptions"):
            basel_traffic_light(bad)
    with pytest.raises(InvalidParameterError, match="integer"):
        basel_traffic_light(True)
