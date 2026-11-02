"""Four VaR models on returns with volatility clustering, and the backtests that judge them.

Run from the repository root:

    uv run python examples/var_backtest.py

The returns are SYNTHETIC: a GARCH(1,1) process (volatility that rises after
large moves and decays slowly) with fat-tailed Student-t shocks, which mimics
two well-known features of daily equity returns. Historical simulation and the
Gaussian model weight a year of history equally, so they react slowly when
volatility jumps and their exceptions cluster. EWMA and filtered historical
simulation follow the volatility. The script prints the backtest of each model
and writes docs/images/var_backtest.png.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantkit.var import (
    METHODS,
    RiskForecast,
    backtest_var,
    basel_traffic_light,
    es_backtest,
    rolling_var_es,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "var_backtest.png"
LEVEL, WINDOW = 0.99, 250
LABELS = {"historical": "historical", "gaussian": "Gaussian", "ewma": "EWMA", "fhs": "FHS"}
COLORS = {
    "historical": "tab:red",
    "gaussian": "tab:orange",
    "ewma": "tab:green",
    "fhs": "tab:blue",
}


def simulate_garch_returns(n_days: int, seed: int) -> pd.Series:
    """GARCH(1,1) returns: sigma2[t] = omega + a r[t-1]^2 + b sigma2[t-1], t(5) shocks."""
    a, b, long_run_vol = 0.08, 0.90, 0.01
    omega = long_run_vol**2 * (1 - a - b)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(5, n_days) * math.sqrt(3 / 5)  # unit variance
    returns = np.empty(n_days)
    variance = long_run_vol**2
    for t in range(n_days):
        returns[t] = math.sqrt(variance) * shocks[t]
        variance = omega + a * returns[t] ** 2 + b * variance
    return pd.Series(returns, index=pd.bdate_range("2018-01-01", periods=n_days, name="date"))


def summarize(forecast: RiskForecast) -> dict[str, object]:
    result = backtest_var(forecast.exceptions, level=forecast.level)
    light = basel_traffic_light(int(forecast.last(250).exceptions.sum()))
    z2 = es_backtest(forecast, n_simulations=9_999, seed=0)
    return {
        "exceptions": result.n_exceptions,
        "expected": round(result.expected_exceptions, 1),
        "Kupiec p": round(result.unconditional.p_value, 4),
        "independence p": round(result.independence.p_value, 4),
        "cond. coverage p": round(result.conditional.p_value, 4),
        "Basel zone (last 250 days)": f"{light.zone} ({light.n_exceptions})",
        "Z2": round(z2.statistic, 2),
        "Z2 p": round(z2.p_value, 4),
    }


def main() -> None:
    """Forecast with all four methods, backtest them on common days, and plot."""
    returns = simulate_garch_returns(1_750, seed=3)
    forecasts = {m: rolling_var_es(returns, method=m, level=LEVEL, window=WINDOW) for m in METHODS}
    n_days = min(len(f) for f in forecasts.values())  # FHS needs two windows of history
    forecasts = {m: f.last(n_days) for m, f in forecasts.items()}
    days = forecasts["fhs"].var.index
    print(
        f"SYNTHETIC GARCH(1,1) returns with t(5) shocks; {LEVEL:.0%} one-day VaR and ES, "
        f"{WINDOW}-day window; backtest over {n_days} days ({days[0].date()} to "
        f"{days[-1].date()})\n"
    )
    table = pd.DataFrame({LABELS[m]: summarize(f) for m, f in forecasts.items()})
    print(table.to_string())

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    realized = forecasts["fhs"].returns
    top.plot(days, realized, color="0.65", lw=0.6, label="daily return")
    for method in ("historical", "fhs"):
        forecast, color = forecasts[method], COLORS[method]
        top.plot(days, -forecast.var, color=color, lw=1.1, label=f"-VaR, {LABELS[method]}")
        hit = forecast.exceptions
        top.scatter(days[hit], realized[hit], color=color, s=14, zorder=3)
    top.set(ylabel="return", title="SYNTHETIC data: 99% one-day VaR and its exceptions (dots)")
    top.yaxis.set_major_formatter("{x:.0%}")
    top.legend(loc="lower left", fontsize=8)
    top.grid(alpha=0.3)

    for method, forecast in forecasts.items():
        bottom.step(
            days,
            forecast.exceptions.cumsum(),
            where="post",
            color=COLORS[method],
            label=LABELS[method],
        )
    expected = (1 - LEVEL) * np.arange(1, n_days + 1)
    bottom.plot(days, expected, color="black", ls="--", lw=1, label="expected (1% of days)")
    bottom.set(ylabel="cumulative exceptions")
    bottom.legend(loc="upper left", fontsize=8, ncol=3)
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110)
    print(f"\nsaved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
