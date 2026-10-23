"""Black-Scholes-Merton prices and analytic Greeks for European options.

With a continuous dividend yield ``q``, the Black-Scholes-Merton (BSM) prices of a
European call ``c`` and put ``p`` are (Hull, *Options, Futures, and Other
Derivatives*, chapters on the BSM model and on options on indices)::

    c = S e^(-qT) N(d1) - K e^(-rT) N(d2)
    p = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    d1 = [ln(S / K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

where ``N`` is the standard normal CDF. ``S e^(-qT)`` is the present value of the
share delivered at expiry and ``K e^(-rT)`` the present value of the strike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantkit.pricing.option import OptionSpec

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function ``N(x)``."""
    # 0.5 * (1 + erf(x / sqrt(2))) cancels catastrophically for very negative x
    # (deep out-of-the-money options); erfc keeps full relative precision there.
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_pdf(x: float) -> float:
    """Standard normal probability density function ``N'(x)``."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def d1_d2(option: OptionSpec) -> tuple[float, float]:
    """Return the BSM terms ``(d1, d2)`` for ``option``."""
    sigma_sqrt_t = option.vol * math.sqrt(option.maturity)
    carry = option.rate - option.dividend_yield + 0.5 * option.vol**2
    d1 = (math.log(option.spot / option.strike) + carry * option.maturity) / sigma_sqrt_t
    return d1, d1 - sigma_sqrt_t


def bsm_price(option: OptionSpec) -> float:
    """Black-Scholes-Merton price of a European call or put.

    Args:
        option: Contract and market inputs. ``option.kind`` selects call or put.

    Returns:
        The arbitrage-free price under geometric Brownian motion with constant
        rate, volatility and dividend yield.

    Example:
        Hull's worked example (S=42, K=40, r=10%, sigma=20%, T=0.5 years):

        >>> call = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5,
        ...                   kind="call")
        >>> round(bsm_price(call), 2)
        4.76
    """
    d1, d2 = d1_d2(option)
    pv_spot = option.spot * math.exp(-option.dividend_yield * option.maturity)
    pv_strike = option.strike * math.exp(-option.rate * option.maturity)
    if option.kind == "call":
        return pv_spot * norm_cdf(d1) - pv_strike * norm_cdf(d2)
    return pv_strike * norm_cdf(-d2) - pv_spot * norm_cdf(-d1)


@dataclass(frozen=True, slots=True)
class Greeks:
    """Sensitivities of an option price to its inputs.

    Units follow the textbook convention (no rescaling to "per 1%" or "per day"),
    which keeps every Greek an exact partial derivative.

    Attributes:
        delta: ``dV/dS``.
        gamma: ``d2V/dS2``.
        vega: ``dV/dsigma`` per unit of volatility (1.00 = 100 vol points).
        theta: ``dV/dt = -dV/dT`` per year of calendar time (usually negative).
        rho: ``dV/dr`` per unit change in the rate (1.00 = 100%).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bsm_greeks(option: OptionSpec) -> Greeks:
    """Analytic Black-Scholes-Merton Greeks with a continuous dividend yield.

    With ``n`` the standard normal density, ``Dq = e^(-qT)`` and ``Dr = e^(-rT)``
    (Hull, chapter "The Greek Letters")::

        delta_call = Dq N(d1)                delta_put = -Dq N(-d1)
        gamma      = Dq n(d1) / (S sigma sqrt(T))
        vega       = S Dq n(d1) sqrt(T)
        theta_call = -S Dq n(d1) sigma / (2 sqrt(T)) + q S Dq N(d1) - r K Dr N(d2)
        theta_put  = -S Dq n(d1) sigma / (2 sqrt(T)) - q S Dq N(-d1) + r K Dr N(-d2)
        rho_call   = K T Dr N(d2)            rho_put = -K T Dr N(-d2)

    Args:
        option: Contract and market inputs.

    Returns:
        The five Greeks; see :class:`Greeks` for units.
    """
    s, k, t = option.spot, option.strike, option.maturity
    r, q, sigma = option.rate, option.dividend_yield, option.vol
    d1, d2 = d1_d2(option)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    density = norm_pdf(d1)

    gamma = disc_q * density / (s * sigma * sqrt_t)
    vega = s * disc_q * density * sqrt_t
    # Time decay of the optionality itself; identical for calls and puts.
    decay = -s * disc_q * density * sigma / (2.0 * sqrt_t)

    if option.kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = decay + q * s * disc_q * norm_cdf(d1) - r * k * disc_r * norm_cdf(d2)
        rho = k * t * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = decay - q * s * disc_q * norm_cdf(-d1) + r * k * disc_r * norm_cdf(-d2)
        rho = -k * t * disc_r * norm_cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
