"""Monte Carlo pricing of European options under geometric Brownian motion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantkit._validation import FloatArray, require_int
from quantkit.exceptions import InvalidParameterError
from quantkit.pricing.option import OptionSpec

Z_95 = 1.959963984540054
"""Two-sided 95% quantile of the standard normal distribution, N^-1(0.975)."""

SeedLike = int | np.random.Generator | None
"""Anything accepted by :func:`numpy.random.default_rng`."""


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """A Monte Carlo price estimate together with its sampling uncertainty.

    Attributes:
        price: Sample mean of the discounted payoffs.
        std_error: Standard error of ``price`` (sample std / sqrt(samples)).
        ci_low: Lower end of the 95% confidence interval.
        ci_high: Upper end of the 95% confidence interval.
        n_paths: Number of simulated paths (each antithetic pair counts as two).
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int

    @classmethod
    def from_samples(cls, samples: FloatArray, n_paths: int) -> MonteCarloResult:
        """Summarize i.i.d. samples whose mean estimates the price.

        The 95% interval ``price +/- 1.96 * std_error`` relies on the central limit
        theorem, which is accurate for the sample sizes used in practice.
        """
        price = float(samples.mean())
        std_error = float(samples.std(ddof=1) / math.sqrt(samples.size))
        return cls(
            price=price,
            std_error=std_error,
            ci_low=price - Z_95 * std_error,
            ci_high=price + Z_95 * std_error,
            n_paths=n_paths,
        )


def monte_carlo_price(
    option: OptionSpec,
    *,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: SeedLike = None,
) -> MonteCarloResult:
    """Estimate a European option price by simulating the terminal stock price.

    Under the risk-neutral measure, geometric Brownian motion has the exact
    solution::

        S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)

    so a European payoff needs only one normal draw per path; there is no
    time-discretization error, only sampling error. The price estimate is the
    average discounted payoff ``e^(-rT) payoff(S_T)``.

    With ``antithetic=True`` each draw ``Z`` is paired with ``-Z`` and the two
    payoffs are averaged into one sample. Because a call or put payoff is
    monotone in ``Z``, the two halves of a pair are negatively correlated, so the
    pair average has lower variance than two independent draws (Glasserman 2003,
    section 4.2). The standard error is computed from the pair averages, which
    are independent of each other.

    Args:
        option: Contract and market inputs.
        n_paths: Total number of simulated terminal prices (even if antithetic).
        antithetic: Use antithetic variates.
        seed: Seed or :class:`numpy.random.Generator`, for reproducible results.

    Returns:
        The estimate, its standard error and a 95% confidence interval.
    """
    n_paths = require_int("n_paths", n_paths, minimum=4)
    if antithetic and n_paths % 2:
        raise InvalidParameterError(f"n_paths must be even when antithetic=True, got {n_paths}")
    rng = np.random.default_rng(seed)

    t = option.maturity
    drift = (option.rate - option.dividend_yield - 0.5 * option.vol**2) * t
    diffusion = option.vol * math.sqrt(t)
    discount = math.exp(-option.rate * t)

    def discounted_payoff(z: FloatArray) -> FloatArray:
        terminal = option.spot * np.exp(drift + diffusion * z)
        return discount * option.payoff(terminal)

    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        samples = 0.5 * (discounted_payoff(z) + discounted_payoff(-z))
    else:
        samples = discounted_payoff(rng.standard_normal(n_paths))
    return MonteCarloResult.from_samples(samples, n_paths)
