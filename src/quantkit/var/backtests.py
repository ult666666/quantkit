"""Exception backtests of VaR forecasts: Kupiec, Christoffersen and conditional coverage.

An *exception* is a day whose loss exceeded that day's VaR forecast. If the
forecasts are correct at confidence level ``c``, exceptions are independent
Bernoulli trials with probability ``p = 1 - c``. Three likelihood-ratio tests
check this (Christoffersen 1998):

* **Unconditional coverage** (Kupiec 1995, proportion of failures): is the
  exception rate consistent with ``p``? With ``x`` exceptions in ``T`` days and
  ``pi = x / T``::

      LR_uc = -2 log[(1 - p)^(T - x) p^x / ((1 - pi)^(T - x) pi^x)]  ~ chi2(1)

* **Independence** (Christoffersen 1998): does an exception today change the
  chance of one tomorrow? With ``n_ij`` the number of days in state ``i``
  followed by a day in state ``j`` (1 = exception), the alternative is a
  first-order Markov chain with ``pi_01 = n01 / (n00 + n01)`` and
  ``pi_11 = n11 / (n10 + n11)``; the null is one common ``pi``::

      LR_ind = -2 log[L(pi) / L(pi_01, pi_11)]  ~ chi2(1)

* **Conditional coverage** (Christoffersen 1998): the Markov chain against
  independent trials with probability exactly ``p``::

      LR_cc = -2 log[L(p) / L(pi_01, pi_11)]  ~ chi2(2)

  ``LR_ind`` and ``LR_cc`` use the ``T - 1`` transitions, so ``LR_cc`` equals
  ``LR_ind`` plus the Kupiec statistic of days ``2..T``.

Likelihood terms use ``0 log 0 = 0``, so every statistic stays finite with no
exceptions, with only exceptions, or with no pair of consecutive exceptions.

P-values come from the chi-squared limits, which are rough when few exceptions
are expected: at 99% over 250 days a correct model expects only 2.5.
:func:`null_rejection_rates` computes how often each test really rejects a
correct model at a given sample size, by enumerating the exact distribution of
the transition counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import binom

from quantkit._validation import FloatArray, as_1d_array, require_int, require_level
from quantkit.exceptions import InvalidParameterError

Counts = int | npt.NDArray[np.int64]
"""A count, or an array of counts for many samples at once."""

# null_rejection_rates ignores exception counts whose total probability is below this.
_NEGLIGIBLE_MASS = 1e-15
# Largest dynamic program (days x states) null_rejection_rates will run, about a second.
_MAX_ENUMERATION_WORK = 500_000_000


@dataclass(frozen=True, slots=True)
class LikelihoodRatioTest:
    """A likelihood-ratio statistic and its asymptotic p-value.

    Attributes:
        statistic: ``-2 log(L_null / L_alternative)``; never negative.
        dof: Degrees of freedom of the limiting chi-squared distribution.
        p_value: ``P(chi2(dof) >= statistic)``.
    """

    statistic: float
    dof: int
    p_value: float


@dataclass(frozen=True, slots=True)
class TransitionCounts:
    """Counts of consecutive day pairs; ``n01`` counts a quiet day followed by an exception.

    Attributes:
        n00: No exception, then no exception.
        n01: No exception, then an exception.
        n10: An exception, then no exception.
        n11: An exception, then another exception.
    """

    n00: int
    n01: int
    n10: int
    n11: int


@dataclass(frozen=True, slots=True)
class VaRBacktest:
    """Exception counts and the three coverage tests for one series of VaR forecasts.

    Attributes:
        level: VaR confidence level of the forecasts.
        n_observations: Number of forecast days ``T``.
        n_exceptions: Number of exceptions ``x``.
        transitions: Consecutive-day transition counts.
        unconditional: Kupiec's proportion-of-failures test (LR_uc).
        independence: Christoffersen's independence test (LR_ind).
        conditional: Christoffersen's conditional coverage test (LR_cc).
    """

    level: float
    n_observations: int
    n_exceptions: int
    transitions: TransitionCounts
    unconditional: LikelihoodRatioTest
    independence: LikelihoodRatioTest
    conditional: LikelihoodRatioTest

    @property
    def expected_exceptions(self) -> float:
        """Exceptions a correct model expects: ``T * (1 - level)``."""
        return self.n_observations * (1.0 - self.level)

    @property
    def exception_rate(self) -> float:
        """Observed exception rate ``x / T``."""
        return self.n_exceptions / self.n_observations


@dataclass(frozen=True, slots=True)
class RejectionRates:
    """How often each coverage test rejects a correct VaR model.

    Attributes:
        unconditional: Rejection probability of Kupiec's LR_uc.
        independence: Rejection probability of Christoffersen's LR_ind.
        conditional: Rejection probability of the conditional coverage LR_cc.
    """

    unconditional: float
    independence: float
    conditional: float


def var_exceptions(
    returns: pd.Series | npt.ArrayLike, var: pd.Series | npt.ArrayLike
) -> pd.Series:
    """Flag the days on which the loss exceeded the VaR forecast: ``return < -var``.

    Args:
        returns: Realized returns.
        var: VaR forecasts for the same days, as positive loss fractions. Two
            Series must share their index; other inputs must have equal lengths.

    Returns:
        A boolean Series (with the index of ``returns`` when it is a Series).
    """
    r = as_1d_array("returns", returns)
    v = as_1d_array("var", var)
    if r.shape != v.shape:
        raise InvalidParameterError(
            f"returns and var must have the same length, got {r.size} and {v.size}"
        )
    if (
        isinstance(returns, pd.Series)
        and isinstance(var, pd.Series)
        and not returns.index.equals(var.index)
    ):
        raise InvalidParameterError("returns and var must have the same index")
    index = returns.index if isinstance(returns, pd.Series) else None
    return pd.Series(r < -v, index=index, name="exception")


def kupiec_test(exceptions: pd.Series | npt.ArrayLike, *, level: float) -> LikelihoodRatioTest:
    """Kupiec's proportion-of-failures test of unconditional coverage (LR_uc).

    The test is two-sided: too few exceptions also count against the model,
    because they mean the VaR is set too high. With no exceptions the
    statistic is ``-2 T log(level)``, so a 99% model with zero exceptions in 250
    days is rejected at 5%.

    Args:
        exceptions: Exception indicators (booleans or 0/1), one per day.
        level: VaR confidence level of the forecasts.
    """
    hits = _hits(exceptions, min_length=1)
    p = 1.0 - require_level("level", level)
    n_one = int(hits.sum())
    statistic = float(_uc_statistic(hits.size - n_one, n_one, p))
    return LikelihoodRatioTest(statistic, 1, _chi2_sf(statistic, 1))


def transition_counts(exceptions: pd.Series | npt.ArrayLike) -> TransitionCounts:
    """Count the four kinds of consecutive day pairs in an exception series."""
    hits = _hits(exceptions, min_length=2)
    before, after = hits[:-1], hits[1:]
    n11 = int((before & after).sum())
    n10 = int((before & ~after).sum())
    n01 = int((~before & after).sum())
    return TransitionCounts(n00=before.size - n11 - n10 - n01, n01=n01, n10=n10, n11=n11)


def christoffersen_test(exceptions: pd.Series | npt.ArrayLike) -> LikelihoodRatioTest:
    """Christoffersen's test that exceptions are independent from day to day (LR_ind).

    With no exceptions, or no quiet days, there is nothing to compare: the
    statistic is 0 and the p-value 1.

    Args:
        exceptions: Exception indicators (booleans or 0/1), at least two days.
    """
    c = transition_counts(exceptions)
    statistic = float(_ind_statistic(c.n00, c.n01, c.n10, c.n11))
    return LikelihoodRatioTest(statistic, 1, _chi2_sf(statistic, 1))


def conditional_coverage_test(
    exceptions: pd.Series | npt.ArrayLike, *, level: float
) -> LikelihoodRatioTest:
    """Christoffersen's joint test of coverage and independence (LR_cc).

    Args:
        exceptions: Exception indicators (booleans or 0/1), at least two days.
        level: VaR confidence level of the forecasts.
    """
    c = transition_counts(exceptions)
    p = 1.0 - require_level("level", level)
    statistic = float(_cc_statistic(c.n00, c.n01, c.n10, c.n11, p))
    return LikelihoodRatioTest(statistic, 2, _chi2_sf(statistic, 2))


def backtest_var(exceptions: pd.Series | npt.ArrayLike, *, level: float) -> VaRBacktest:
    """Count exceptions and run all three coverage tests.

    Args:
        exceptions: Exception indicators (booleans or 0/1), at least two days;
            for example :attr:`RiskForecast.exceptions` or :func:`var_exceptions`.
        level: VaR confidence level of the forecasts.
    """
    hits = _hits(exceptions, min_length=2)
    level = require_level("level", level)
    return VaRBacktest(
        level=level,
        n_observations=hits.size,
        n_exceptions=int(hits.sum()),
        transitions=transition_counts(hits),
        unconditional=kupiec_test(hits, level=level),
        independence=christoffersen_test(hits),
        conditional=conditional_coverage_test(hits, level=level),
    )


def null_rejection_rates(
    n_observations: int, *, level: float, significance: float = 0.05
) -> RejectionRates:
    """Exact probability that each test rejects a correct model, at its asymptotic cutoff.

    Under a correct model the exceptions are independent Bernoulli(p) trials, so
    the three statistics have exact distributions that the chi-squared limits
    only approximate. A dynamic program over the days tracks the joint
    distribution of the first and last day's state, the number of exceptions and
    the number of consecutive exception pairs; those four numbers determine the
    transition counts and so every statistic. Exception counts with a total
    probability below 1e-15 are left out.

    For example, at 99% over 250 days the 5% Kupiec test rejects a correct model
    far more often than 5%: no exceptions at all, which happens with probability
    0.99^250 = 8.1%, already gives ``LR_uc = 5.0 > 3.84``.

    Args:
        n_observations: Number of backtest days ``T`` (at least 2).
        level: VaR confidence level, e.g. 0.99.
        significance: Nominal level of the tests, e.g. 0.05.

    Returns:
        The probability that each test's asymptotic p-value falls below
        ``significance`` when the model is correct.

    Raises:
        InvalidParameterError: For invalid inputs, or a sample so long (with so
            many expected exceptions) that the enumeration would be too slow.
    """
    n = require_int("n_observations", n_observations, minimum=2)
    p = 1.0 - require_level("level", level)
    significance = require_level("significance", significance)
    max_ones = min(n, int(binom.isf(_NEGLIGIBLE_MASS, n, p)) + 1)
    if n * (max_ones + 1) ** 2 > _MAX_ENUMERATION_WORK:
        raise InvalidParameterError(
            f"enumerating {n} days with up to {max_ones} exceptions is too slow; "
            "simulate i.i.d. Bernoulli exceptions instead"
        )

    # prob[first, last, ones, pairs]: distribution after the days processed so far.
    prob = np.zeros((2, 2, max_ones + 1, max_ones + 1))
    prob[0, 0, 0, 0] = 1.0 - p
    prob[1, 1, 1, 0] = p
    for _ in range(n - 1):
        step = np.zeros_like(prob)
        step[:, 0] = (prob[:, 0] + prob[:, 1]) * (1.0 - p)  # a quiet day
        step[:, 1, 1:, :] = prob[:, 0, :-1, :] * p  # an exception after a quiet day
        step[:, 1, 1:, 1:] += prob[:, 1, :-1, :-1] * p  # an exception after an exception
        prob = step

    first, last, ones, n11 = (axis.ravel() for axis in np.indices(prob.shape))
    weight = prob.ravel()
    # Exceptions on days 1..T-1 and 2..T give the transition counts.
    n10 = ones - last - n11
    n01 = ones - first - n11
    n00 = (n - 1) - n01 - n10 - n11
    possible = (weight > 0) & (np.minimum(np.minimum(n00, n01), n10) >= 0)
    first, ones, n00, n01, n10, n11, weight = (
        a[possible] for a in (first, ones, n00, n01, n10, n11, weight)
    )
    cutoff_1 = NormalDist().inv_cdf(1.0 - significance / 2.0) ** 2  # chi2(1) quantile
    cutoff_2 = -2.0 * math.log(significance)  # chi2(2) quantile
    return RejectionRates(
        unconditional=float(weight[_uc_statistic(n - ones, ones, p) > cutoff_1].sum()),
        independence=float(weight[_ind_statistic(n00, n01, n10, n11) > cutoff_1].sum()),
        conditional=float(weight[_cc_statistic(n00, n01, n10, n11, p) > cutoff_2].sum()),
    )


def _hits(exceptions: pd.Series | npt.ArrayLike, *, min_length: int) -> npt.NDArray[np.bool_]:
    """Exception indicators as a boolean array, accepting booleans or 0/1 numbers."""
    array = np.asarray(exceptions)
    if array.dtype != np.bool_:
        values = as_1d_array("exceptions", array)
        if not np.isin(values, (0.0, 1.0)).all():
            raise InvalidParameterError("exceptions must be booleans or 0/1 values")
        array = values == 1.0
    if array.ndim != 1:
        raise InvalidParameterError(f"exceptions must be one-dimensional, got shape {array.shape}")
    if array.size < min_length:
        raise InvalidParameterError(
            f"exceptions need at least {min_length} observations, got {array.size}"
        )
    return array


def _xlogy(count: Counts, probability: float | FloatArray) -> FloatArray:
    """``count * log(probability)``, with ``0 * log(0) = 0``."""
    count_array = np.asarray(count, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = count_array * np.log(probability)
    result: FloatArray = np.where(count_array > 0, terms, 0.0)
    return result


def _loglik(n_zero: Counts, n_one: Counts, p: float) -> FloatArray:
    """Log-likelihood of ``n_zero`` zeros and ``n_one`` ones from Bernoulli(p) trials."""
    return _xlogy(n_zero, 1.0 - p) + _xlogy(n_one, p)


def _max_loglik(n_zero: Counts, n_one: Counts) -> FloatArray:
    """The Bernoulli log-likelihood at its maximum, ``p = n_one / (n_zero + n_one)``.

    With no trials at all the likelihood is an empty product: log-likelihood 0.
    """
    total = np.asarray(n_zero + n_one, dtype=np.float64)
    trials = np.where(total > 0, total, 1.0)
    return _xlogy(n_zero, n_zero / trials) + _xlogy(n_one, n_one / trials)


def _likelihood_ratio(null_loglik: FloatArray, alternative_loglik: FloatArray) -> FloatArray:
    # The alternative nests the null, so the ratio is >= 0 up to round-off.
    ratio: FloatArray = np.maximum(0.0, 2.0 * (alternative_loglik - null_loglik))
    return ratio


def _uc_statistic(n_zero: Counts, n_one: Counts, p: float) -> FloatArray:
    return _likelihood_ratio(_loglik(n_zero, n_one, p), _max_loglik(n_zero, n_one))


def _markov_loglik(n00: Counts, n01: Counts, n10: Counts, n11: Counts) -> FloatArray:
    """Maximized log-likelihood of a first-order Markov chain with these transitions."""
    return _max_loglik(n00, n01) + _max_loglik(n10, n11)


def _ind_statistic(n00: Counts, n01: Counts, n10: Counts, n11: Counts) -> FloatArray:
    iid = _max_loglik(n00 + n10, n01 + n11)
    return _likelihood_ratio(iid, _markov_loglik(n00, n01, n10, n11))


def _cc_statistic(n00: Counts, n01: Counts, n10: Counts, n11: Counts, p: float) -> FloatArray:
    null = _loglik(n00 + n10, n01 + n11, p)
    return _likelihood_ratio(null, _markov_loglik(n00, n01, n10, n11))


def _chi2_sf(statistic: float, dof: int) -> float:
    """Upper tail of the chi-squared distribution with 1 or 2 degrees of freedom.

    chi2(1) is the square of a standard normal, so ``P(X > x) = erfc(sqrt(x / 2))``;
    chi2(2) is exponential with mean 2, so ``P(X > x) = exp(-x / 2)``.
    """
    if dof == 1:
        return math.erfc(math.sqrt(statistic / 2.0))
    return math.exp(-statistic / 2.0)
