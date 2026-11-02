"""VaR model validation: rolling VaR and ES forecasts and the backtests that judge them.

* :func:`rolling_var_es` produces one-day VaR and expected shortfall forecasts by
  historical simulation, a Gaussian model, EWMA (RiskMetrics) or filtered
  historical simulation, each from a trailing window that never includes the
  return being forecast.
* :func:`backtest_var` counts the exceptions (days whose loss exceeded VaR) and
  runs Kupiec's and Christoffersen's likelihood-ratio tests;
  :func:`null_rejection_rates` gives their exact size at a given sample length.
* :func:`basel_traffic_light` maps 250 days of 99% exceptions to the Basel
  green, yellow and red zones and their capital plus factors.
* :func:`es_backtest` tests expected shortfall with Acerbi and Székely's ``Z2``
  statistic against its simulated null distribution.

Returns, VaR and ES are decimals; VaR and ES are positive loss fractions.
"""

from quantkit.var.backtests import (
    LikelihoodRatioTest,
    RejectionRates,
    TransitionCounts,
    VaRBacktest,
    backtest_var,
    christoffersen_test,
    conditional_coverage_test,
    kupiec_test,
    null_rejection_rates,
    transition_counts,
    var_exceptions,
)
from quantkit.var.es_backtest import ESBacktest, acerbi_szekely_z2, es_backtest
from quantkit.var.forecasts import (
    METHODS,
    RISKMETRICS_DECAY,
    Method,
    RiskForecast,
    ewma_weights,
    rolling_var_es,
)
from quantkit.var.traffic_light import (
    BASEL_DAYS,
    BASEL_LEVEL,
    TrafficLight,
    Zone,
    basel_traffic_light,
)

__all__ = [
    "BASEL_DAYS",
    "BASEL_LEVEL",
    "METHODS",
    "RISKMETRICS_DECAY",
    "ESBacktest",
    "LikelihoodRatioTest",
    "Method",
    "RejectionRates",
    "RiskForecast",
    "TrafficLight",
    "TransitionCounts",
    "VaRBacktest",
    "Zone",
    "acerbi_szekely_z2",
    "backtest_var",
    "basel_traffic_light",
    "christoffersen_test",
    "conditional_coverage_test",
    "es_backtest",
    "ewma_weights",
    "kupiec_test",
    "null_rejection_rates",
    "rolling_var_es",
    "transition_counts",
    "var_exceptions",
]
