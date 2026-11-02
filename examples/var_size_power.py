"""How often do the VaR and ES backtests reject a correct model, and a wrong one?

Run from the repository root:

    uv run python examples/var_size_power.py

All data are SYNTHETIC. The script simulates many independent return paths,
forecasts VaR and ES on each with a 250-day window, runs the 5% tests on the
last 250 or 1,000 days, and prints the share of paths each test rejects.

* Size: returns drawn from the EWMA (RiskMetrics) model itself, so the EWMA
  forecast is the true conditional distribution and every rejection is a false
  alarm. The exact rejection probability of each likelihood-ratio test comes
  from quantkit.var.null_rejection_rates. The Z2 Monte Carlo test with 199
  simulations has an exact 5% size, because 5% of 199 + 1 is a whole number.
* Power: Gaussian forecasts on Student-t returns with 4 degrees of freedom,
  whose tails are fatter than the model assumes, next to the same forecasts on
  normal returns (the Gaussian model's own assumption, with estimation error).
"""

import math
from collections.abc import Callable

import numpy as np

from quantkit.var import (
    Method,
    backtest_var,
    es_backtest,
    ewma_weights,
    null_rejection_rates,
    rolling_var_es,
)

N_PATHS = 1_000
WINDOW = 250
Z2_SIMULATIONS = 199  # 0.05 * (199 + 1) = 10, so 5% is an exact Monte Carlo cutoff
Sampler = Callable[[np.random.Generator, int], np.ndarray]


def ewma_model(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Returns from the EWMA model: r[t] = sigma[t] z[t], sigma from the past 250 returns."""
    weights = ewma_weights(WINDOW)
    paths = np.empty((N_PATHS, n_days))
    paths[:, :WINDOW] = 0.01 * rng.standard_normal((N_PATHS, WINDOW))
    for t in range(WINDOW, n_days):
        sigma = np.sqrt(np.square(paths[:, t - WINDOW : t]) @ weights)
        paths[:, t] = sigma * rng.standard_normal(N_PATHS)
    return paths


def normal(rng: np.random.Generator, n_days: int) -> np.ndarray:
    return 0.01 * rng.standard_normal((N_PATHS, n_days))


def student_t4(rng: np.random.Generator, n_days: int) -> np.ndarray:
    return 0.01 * rng.standard_t(4, (N_PATHS, n_days)) * math.sqrt(2 / 4)  # 1% daily vol


def rejection_rates(
    sampler: Sampler, method: Method, level: float, n_test: int, rng: np.random.Generator
) -> list[float]:
    """Share of paths rejected by LR_uc, LR_ind, LR_cc and Z2 at 5%, and the exception rate."""
    rejected = np.zeros(4)
    exceptions = 0
    for path in sampler(rng, WINDOW + n_test):
        forecast = rolling_var_es(path, method=method, level=level, window=WINDOW)
        result = backtest_var(forecast.exceptions, level=level)
        z2 = es_backtest(forecast, n_simulations=Z2_SIMULATIONS, seed=rng)
        rejected += [
            result.unconditional.p_value < 0.05,
            result.independence.p_value < 0.05,
            result.conditional.p_value < 0.05,
            z2.p_value <= 0.05,
        ]
        exceptions += result.n_exceptions
    return [*(rejected / N_PATHS), exceptions / (N_PATHS * n_test)]


def row(label: str, values: list[float]) -> str:
    cells = "".join(f"{v:>9.1%}" for v in values[:4])
    rate = f"{values[4]:>11.2%}" if len(values) > 4 else ""
    return f"{label:<46}{cells}{rate}"


def main() -> None:
    """Run the size and power simulations and print one table."""
    rng = np.random.default_rng(2026)
    print(
        f"Share of {N_PATHS:,} SYNTHETIC paths rejected by each 5% test "
        f"(Monte Carlo standard error at most {0.5 / math.sqrt(N_PATHS):.1%})\n"
    )
    print(f"{'':<46}{'LR_uc':>9}{'LR_ind':>9}{'LR_cc':>9}{'Z2':>9}{'exceptions':>11}")
    print("size: EWMA forecasts on returns from the EWMA model, 99%")
    for n_test in (250, 1_000):
        measured = rejection_rates(ewma_model, "ewma", 0.99, n_test, rng)
        exact = null_rejection_rates(n_test, level=0.99)
        print(row(f"  {n_test:,} days, measured", measured))
        print(row(f"  {n_test:,} days, exact", [*vars_of(exact), 0.05]))
    for level in (0.99, 0.975):
        print(f"power: Gaussian forecasts, {100 * level:g}%")
        tests = (250, 1_000) if level == 0.99 else (1_000,)
        for n_test in tests:
            for name, sampler in (("normal", normal), ("Student-t(4)", student_t4)):
                values = rejection_rates(sampler, "gaussian", level, n_test, rng)
                print(row(f"  {name} returns, {n_test:,} days", values))


def vars_of(rates: object) -> list[float]:
    """The three exact rejection probabilities, in table order."""
    return [getattr(rates, name) for name in ("unconditional", "independence", "conditional")]


if __name__ == "__main__":
    main()
