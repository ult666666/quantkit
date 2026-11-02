"""Acerbi and Székely's Z2 backtest of expected shortfall forecasts.

Expected shortfall (ES) is the mean loss on the days beyond VaR. Acerbi and
Székely (2014) test ES forecasts directly with their second statistic::

    Z2 = sum_t r[t] I[t] / (T alpha ES[t]) + 1,    I[t] = 1 if r[t] < -VaR[t] else 0

where ``alpha = 1 - level`` is the tail probability and ``T`` the number of
days. If each day's forecast distribution is right and continuous, then
``E[r I] = -alpha ES`` on every day, so ``E[Z2] = 0``. When ES is
underestimated, the losses beyond VaR are larger or more frequent than
forecast, and ``Z2`` is negative. The test is one-sided: small values reject.

``Z2`` has no standard distribution. As the paper prescribes, its null
distribution is simulated: draw many return paths from the forecast
distributions themselves (:meth:`~quantkit.var.RiskForecast.simulate`), compute
``Z2`` on each, and locate the observed value among them. The p-value counts the
observed path as one of the draws, ``(1 + #{Z_sim <= Z_obs}) / (N + 1)``, which
makes it a Monte Carlo test with the correct size at any sample length
(Davison & Hinkley 1997, chapter 4). Ties, such as ``Z2 = 1`` on paths without
exceptions, are counted against the model, which keeps the test conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import FloatArray, SeedLike, as_1d_array, require_int, require_level
from quantkit.exceptions import InvalidParameterError
from quantkit.var.forecasts import RiskForecast

# Simulate at most this many path-days at once to bound memory use (about 16 MB).
_MAX_DRAWS_PER_BATCH = 2_000_000


@dataclass(frozen=True, slots=True)
class ESBacktest:
    """The Z2 statistic and its simulated null distribution.

    Attributes:
        statistic: Observed ``Z2``; 0 is ideal, negative values mean ES was too low.
        p_value: Simulated one-sided p-value, ``P(Z2 <= statistic)`` under the null.
        critical_value: The ``significance`` quantile of the simulated ``Z2``;
            the test rejects when ``statistic`` is below it.
        significance: Significance level of ``critical_value``.
        n_simulations: Number of simulated paths.
    """

    statistic: float
    p_value: float
    critical_value: float
    significance: float
    n_simulations: int


def acerbi_szekely_z2(
    returns: pd.Series | npt.ArrayLike,
    var: pd.Series | npt.ArrayLike,
    es: pd.Series | npt.ArrayLike,
    *,
    level: float,
) -> float:
    """Acerbi and Székely's ``Z2`` statistic for VaR and ES forecasts.

    Args:
        returns: Realized returns.
        var: VaR forecasts for the same days, as positive loss fractions.
        es: ES forecasts for the same days, positive.
        level: Confidence level of the forecasts, e.g. 0.975.

    Returns:
        ``Z2``: 1 if there are no exceptions, about 0 for correct forecasts on
        average, negative when losses beyond VaR exceed the ES forecasts.
    """
    r = as_1d_array("returns", returns)
    v = as_1d_array("var", var)
    e = as_1d_array("es", es)
    if not r.shape == v.shape == e.shape:
        raise InvalidParameterError(
            f"returns, var and es must have the same length, got {r.size}, {v.size} and {e.size}"
        )
    _require_positive_es(e)
    return float(_z2(r[np.newaxis, :], v, e, 1.0 - require_level("level", level))[0])


def es_backtest(
    forecast: RiskForecast,
    *,
    n_simulations: int = 10_000,
    significance: float = 0.05,
    seed: SeedLike = None,
) -> ESBacktest:
    """Test a forecast's ES with ``Z2`` against its own simulated null distribution.

    Args:
        forecast: Rolling forecasts from :func:`~quantkit.var.rolling_var_es`.
        n_simulations: Number of null paths to simulate.
        significance: Level for the reported critical value, e.g. 0.05.
        seed: Seed or :class:`numpy.random.Generator` for reproducibility.

    Returns:
        The observed statistic, its simulated p-value and critical value.
    """
    n_simulations = require_int("n_simulations", n_simulations, minimum=1)
    significance = require_level("significance", significance)
    var = forecast.var.to_numpy()
    es = forecast.es.to_numpy()
    _require_positive_es(es)
    alpha = 1.0 - forecast.level
    observed = float(_z2(forecast.returns.to_numpy()[np.newaxis, :], var, es, alpha)[0])

    rng = np.random.default_rng(seed)
    batch = max(1, _MAX_DRAWS_PER_BATCH // len(forecast))
    simulated = np.concatenate(
        [
            _z2(forecast.simulate(min(batch, n_simulations - start), seed=rng), var, es, alpha)
            for start in range(0, n_simulations, batch)
        ]
    )
    return ESBacktest(
        statistic=observed,
        p_value=(1 + int(np.count_nonzero(simulated <= observed))) / (n_simulations + 1),
        critical_value=float(np.quantile(simulated, significance)),
        significance=significance,
        n_simulations=n_simulations,
    )


def _z2(paths: FloatArray, var: FloatArray, es: FloatArray, alpha: float) -> FloatArray:
    """``Z2`` of every row of ``paths`` (one return path per row)."""
    tail_terms = np.where(paths < -var, paths / es, 0.0)
    z2: FloatArray = tail_terms.sum(axis=1) / (paths.shape[1] * alpha) + 1.0
    return z2


def _require_positive_es(es: FloatArray) -> None:
    if not (es > 0.0).all():
        raise InvalidParameterError(
            "Z2 divides by the ES forecasts, so every ES forecast must be positive"
        )
