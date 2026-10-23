"""Implied volatility: invert the Black-Scholes-Merton formula with Brent's method."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from quantkit._validation import require_finite, require_positive
from quantkit.exceptions import ArbitrageBoundError, ConvergenceError, InvalidParameterError
from quantkit.pricing.black_scholes import bsm_price
from quantkit.pricing.option import OptionKind, OptionSpec

VOL_SEARCH_INTERVAL = (1e-6, 10.0)
"""Volatilities searched by :func:`implied_vol`: 0.0001% to 1000% per year."""


def no_arbitrage_bounds(
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    kind: OptionKind,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Model-free bounds that the price of a European option must respect.

    With ``PV(S) = S e^(-qT)`` and ``PV(K) = K e^(-rT)`` (Hull, chapter
    "Properties of Stock Options")::

        max(PV(S) - PV(K), 0) <= call <= PV(S)
        max(PV(K) - PV(S), 0) <= put  <= PV(K)

    A price outside these bounds offers a static arbitrage. BSM prices tend to
    the lower bound as volatility goes to zero, to the upper bound as it goes to
    infinity, and increase strictly in between, so an implied volatility exists
    exactly when a price lies strictly inside the bounds.

    Returns:
        ``(lower, upper)``.
    """
    require_positive("spot", spot)
    require_positive("strike", strike)
    require_positive("maturity", maturity)
    require_finite("rate", rate)
    require_finite("dividend_yield", dividend_yield)
    pv_spot = spot * math.exp(-dividend_yield * maturity)
    pv_strike = strike * math.exp(-rate * maturity)
    if kind == "call":
        return max(pv_spot - pv_strike, 0.0), pv_spot
    if kind == "put":
        return max(pv_strike - pv_spot, 0.0), pv_strike
    raise InvalidParameterError(f"kind must be 'call' or 'put', got {kind!r}")


def implied_vol(
    price: float,
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    kind: OptionKind,
    dividend_yield: float = 0.0,
) -> float:
    """Volatility at which the Black-Scholes-Merton price equals ``price``.

    The pricing error ``f(sigma) = BSM(sigma) - price`` is continuous and strictly
    increasing (vega > 0), so inside the no-arbitrage bounds the root is unique.
    Brent's method (:func:`scipy.optimize.brentq`) keeps the root bracketed at
    every step, combining the guaranteed convergence of bisection with the speed
    of interpolation. Unlike Newton's method it cannot diverge when vega is tiny,
    as it is for deep in- or out-of-the-money options.

    Args:
        price: Observed option price.
        spot: Underlying price.
        strike: Strike price.
        rate: Continuously compounded risk-free rate.
        maturity: Time to expiry in years.
        kind: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield.

    Returns:
        The implied volatility (annualized, as a decimal: 0.2 means 20%).

    Raises:
        ArbitrageBoundError: If ``price`` violates the no-arbitrage bounds, so no
            volatility can reproduce it.
        ConvergenceError: If the implied volatility falls outside
            :data:`VOL_SEARCH_INTERVAL`.

    Example:
        >>> round(implied_vol(4.7594, spot=42, strike=40, rate=0.10, maturity=0.5,
        ...                   kind="call"), 4)
        0.2
    """
    require_finite("price", price)
    lower, upper = no_arbitrage_bounds(
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        kind=kind,
        dividend_yield=dividend_yield,
    )
    if not lower < price < upper:
        raise ArbitrageBoundError(price, lower, upper)

    def pricing_error(vol: float) -> float:
        option = OptionSpec(
            spot=spot,
            strike=strike,
            rate=rate,
            vol=vol,
            maturity=maturity,
            dividend_yield=dividend_yield,
            kind=kind,
        )
        return bsm_price(option) - price

    vol_low, vol_high = VOL_SEARCH_INTERVAL
    if pricing_error(vol_high) < 0.0:
        raise ConvergenceError(
            f"implied volatility exceeds {vol_high:g}; the price is too close to its "
            "upper no-arbitrage bound"
        )
    if pricing_error(vol_low) > 0.0:
        raise ConvergenceError(
            f"implied volatility is below {vol_low:g}; the price is numerically "
            "indistinguishable from its lower no-arbitrage bound"
        )
    return float(brentq(pricing_error, vol_low, vol_high, xtol=1e-12, maxiter=200))
