"""Asian (average-price) options: geometric closed form and arithmetic Monte Carlo.

An Asian option pays off on the average of the underlying's price at ``n``
equally spaced fixing dates ``t_i = i T / n`` (``i = 1..n``) instead of the
price at expiry. For the *arithmetic* average there is no closed form, but the
*geometric* average of lognormal prices is itself lognormal, so its option has a
Black-Scholes-type formula (Kemna & Vorst 1990). The two averages are almost
perfectly correlated, which makes the geometric option an excellent control
variate for pricing the arithmetic one by Monte Carlo.
"""

from __future__ import annotations

import math

import numpy as np

from quantkit._validation import SeedLike, require_int
from quantkit.pricing.black_scholes import norm_cdf
from quantkit.pricing.monte_carlo import MonteCarloResult
from quantkit.pricing.option import OptionSpec


def geometric_asian_price(option: OptionSpec, *, n_fixings: int) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option.

    With ``G = (S(t_1) S(t_2) ... S(t_n))^(1/n)``, ``log G`` is normal under GBM with::

        m = ln S + (r - q - sigma^2 / 2) T (n + 1) / (2n)       (mean)
        v = sigma^2 T (n + 1) (2n + 1) / (6 n^2)                (variance)

    so, with ``d1 = (m - ln K + v) / sqrt(v)`` and ``d2 = d1 - sqrt(v)``::

        call = e^(-rT) [e^(m + v/2) N(d1) - K N(d2)]
        put  = e^(-rT) [K N(-d2) - e^(m + v/2) N(-d1)]

    With a single fixing at expiry (``n = 1``) this is exactly Black-Scholes-Merton.

    Args:
        option: Contract and market inputs; ``option.strike`` applies to the average.
        n_fixings: Number of equally spaced averaging dates, the last at expiry.

    Returns:
        The option price.
    """
    n = require_int("n_fixings", n_fixings, minimum=1)
    s, k, t = option.spot, option.strike, option.maturity
    r, q, sigma = option.rate, option.dividend_yield, option.vol
    mean = math.log(s) + (r - q - 0.5 * sigma**2) * t * (n + 1) / (2 * n)
    variance = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = math.sqrt(variance)
    d1 = (mean - math.log(k) + variance) / sd
    d2 = d1 - sd
    forward = math.exp(mean + 0.5 * variance)  # E[G] under the risk-neutral measure
    discount = math.exp(-r * t)
    if option.kind == "call":
        return discount * (forward * norm_cdf(d1) - k * norm_cdf(d2))
    return discount * (k * norm_cdf(-d2) - forward * norm_cdf(-d1))


def asian_monte_carlo_price(
    option: OptionSpec,
    *,
    n_fixings: int,
    n_paths: int = 100_000,
    control_variate: bool = True,
    seed: SeedLike = None,
) -> MonteCarloResult:
    """Price an arithmetic-average Asian option by Monte Carlo.

    The payoff is ``max(A - K, 0)`` for a call (``max(K - A, 0)`` for a put), with
    ``A`` the arithmetic mean of the prices at the ``n_fixings`` dates. Prices are
    simulated exactly at those dates from GBM's lognormal increments.

    With ``control_variate=True`` each path also yields the geometric-average
    payoff ``X``, whose expectation is known from :func:`geometric_asian_price`.
    The estimator is::

        Y_cv = Y - b (X - E[X]),   b = Cov(Y, X) / Var(X)

    which is unbiased for any fixed ``b`` and has the smallest variance at this
    ``b``; here ``b`` is estimated from the same paths, which introduces a bias
    of order ``1 / n_paths`` that is negligible in practice (Glasserman 2003,
    section 4.1). Because ``X`` and ``Y`` are almost perfectly correlated, the
    standard error typically drops by one to two orders of magnitude.

    Memory use is proportional to ``n_paths * n_fixings``.

    Args:
        option: Contract and market inputs; ``option.strike`` applies to the average.
        n_fixings: Number of equally spaced averaging dates, the last at expiry.
        n_paths: Number of simulated paths.
        control_variate: Use the geometric-average control variate.
        seed: Seed or :class:`numpy.random.Generator`, for reproducible results.

    Returns:
        The estimate, its standard error and a 95% confidence interval.
    """
    n = require_int("n_fixings", n_fixings, minimum=1)
    n_paths = require_int("n_paths", n_paths, minimum=4)
    rng = np.random.default_rng(seed)

    dt = option.maturity / n
    drift = (option.rate - option.dividend_yield - 0.5 * option.vol**2) * dt
    shocks = option.vol * math.sqrt(dt) * rng.standard_normal((n_paths, n))
    log_prices = math.log(option.spot) + np.cumsum(drift + shocks, axis=1)
    discount = math.exp(-option.rate * option.maturity)

    arithmetic = discount * option.payoff(np.exp(log_prices).mean(axis=1))
    if not control_variate:
        return MonteCarloResult.from_samples(arithmetic, n_paths)

    geometric = discount * option.payoff(np.exp(log_prices.mean(axis=1)))
    covariance = np.cov(arithmetic, geometric)
    # If no path pays off, both payoffs are identically zero and b is irrelevant.
    beta = covariance[0, 1] / covariance[1, 1] if covariance[1, 1] > 0 else 0.0
    exact = geometric_asian_price(option, n_fixings=n)
    return MonteCarloResult.from_samples(arithmetic - beta * (geometric - exact), n_paths)
