"""Price data: a seeded synthetic generator and a validating CSV loader.

All price tables in quantkit are *wide*: a :class:`pandas.DataFrame` indexed by a
sorted :class:`pandas.DatetimeIndex` named ``date``, with one column of closing
prices per ticker and no missing values.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import (
    FloatArray,
    SeedLike,
    require_int,
    require_positive,
    validate_prices,
)
from quantkit.exceptions import DataValidationError, InvalidParameterError


def simulate_gbm_prices(
    *,
    tickers: Sequence[str] = ("AAA", "BBB", "CCC"),
    n_days: int = 1260,
    drift: npt.ArrayLike = 0.07,
    vol: npt.ArrayLike = 0.20,
    correlation: npt.ArrayLike = 0.0,
    initial_price: float = 100.0,
    start: str = "2020-01-01",
    seed: SeedLike = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Simulate daily closes of correlated assets following geometric Brownian motion.

    Asset ``i`` follows ``dS_i / S_i = mu_i dt + sigma_i dW_i`` with
    ``corr(dW_i, dW_j) = rho_ij``. Daily log returns are then i.i.d. multivariate
    normal::

        log(S_t / S_{t-1}) ~ N((mu - sigma^2 / 2) dt, Sigma dt),
        Sigma_ij = rho_ij sigma_i sigma_j,   dt = 1 / periods_per_year

    Correlated draws are produced as ``z @ L.T``, where ``z`` holds independent
    standard normals and ``L`` is the Cholesky factor of ``Sigma dt``
    (``L L' = Sigma dt``), so the draws have exactly the target covariance
    (Glasserman 2003, section 2.3).

    The output is synthetic and intended for examples and tests: dates are
    consecutive business days (no holidays) and returns have no fat tails,
    volatility clustering or regime changes.

    Args:
        tickers: Column names, one per simulated asset.
        n_days: Number of rows (including the starting price).
        drift: Annualized expected return ``mu`` per asset (scalar = same for all).
        vol: Annualized volatility ``sigma`` per asset (scalar = same for all).
        correlation: Correlation of the Brownian motions: a full matrix, or one
            number used for every pair.
        initial_price: Price of every asset on the first date.
        start: First date; later dates are consecutive business days.
        seed: Seed or :class:`numpy.random.Generator` for reproducibility.
        periods_per_year: Trading days per year, used to scale ``dt``.

    Returns:
        A wide price table with ``n_days`` rows and one column per ticker.

    Raises:
        InvalidParameterError: For invalid shapes or values, or a correlation
            matrix that is not a valid (positive definite) correlation matrix.
    """
    names = [str(t) for t in tickers]
    if not names or len(set(names)) != len(names):
        raise InvalidParameterError("tickers must be a non-empty sequence of unique names")
    n_assets = len(names)
    n_days = require_int("n_days", n_days, minimum=2)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    initial_price = require_positive("initial_price", initial_price)
    mu = _per_asset("drift", drift, n_assets)
    sigma = _per_asset("vol", vol, n_assets)
    if (sigma <= 0).any():
        raise InvalidParameterError("vol must be positive for every asset")
    corr = _correlation_matrix(correlation, n_assets)

    dt = 1.0 / periods_per_year
    try:
        chol = np.linalg.cholesky(np.outer(sigma, sigma) * corr * dt)
    except np.linalg.LinAlgError as exc:
        raise InvalidParameterError("correlation matrix must be positive definite") from exc

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_days - 1, n_assets)) @ chol.T
    log_returns = (mu - 0.5 * sigma**2) * dt + shocks
    log_paths = np.vstack([np.zeros(n_assets), np.cumsum(log_returns, axis=0)])
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    return pd.DataFrame(initial_price * np.exp(log_paths), index=index, columns=names)


def _per_asset(name: str, value: npt.ArrayLike, n_assets: int) -> FloatArray:
    """Broadcast a scalar to every asset, or check a per-asset sequence."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n_assets, float(array))
    if array.shape != (n_assets,):
        raise InvalidParameterError(
            f"{name} must be a scalar or have one value per ticker ({n_assets}), "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidParameterError(f"{name} must be finite")
    return array


def _correlation_matrix(correlation: npt.ArrayLike, n_assets: int) -> FloatArray:
    """Build (from a scalar) or check a correlation matrix."""
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim == 0:
        matrix = np.full((n_assets, n_assets), float(matrix))
        np.fill_diagonal(matrix, 1.0)
    if matrix.shape != (n_assets, n_assets):
        raise InvalidParameterError(
            f"correlation matrix must have shape ({n_assets}, {n_assets}), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise InvalidParameterError("correlation matrix must be finite and symmetric")
    if not np.allclose(np.diag(matrix), 1.0):
        raise InvalidParameterError("correlation matrix must have ones on the diagonal")
    return matrix


def load_prices_csv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load daily closing prices from a CSV file into a validated wide table.

    Two layouts are accepted, detected from the header (case-insensitive):

    * **long**: columns ``date``, ``ticker`` and ``close``, one row per date and
      ticker in any order;
    * **wide**: a ``date`` column followed by one numeric column per ticker.

    Dates must be ISO 8601 (``YYYY-MM-DD``). Rows may be in any order; they are
    sorted by date.

    Args:
        path: Path to the CSV file.

    Returns:
        A float64 wide table indexed by a sorted DatetimeIndex named ``date``.

    Raises:
        DataValidationError: If the file cannot be parsed or the data has missing
            or non-positive prices, duplicate dates (or date-ticker pairs) or
            unparseable dates.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        raw = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"could not parse {os.fspath(path)!r} as CSV: {exc}") from exc

    # read_csv always produces string labels; map normalized names to the originals.
    header = {str(column).strip().lower(): str(column) for column in raw.columns}
    if "date" not in header:
        raise DataValidationError("CSV needs a 'date' column")
    if "ticker" in header and "close" in header:
        wide = _pivot_long(raw, header["date"], header["ticker"], header["close"])
    else:
        wide = raw.set_index(header["date"])

    try:
        # Requiring ISO 8601 (e.g. 2024-01-31) avoids silently misreading
        # ambiguous dates such as 01/02/2024 as January 2 or February 1.
        wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index, format="ISO8601"), name="date")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"could not parse dates: {exc}") from exc
    wide.columns = pd.Index([str(column).strip() for column in wide.columns])
    return validate_prices(wide.sort_index(), require_datetime_index=True)


def _pivot_long(raw: pd.DataFrame, date: str, ticker: str, close: str) -> pd.DataFrame:
    """Turn long rows (date, ticker, close) into a wide table, rejecting duplicates."""
    duplicated = raw.duplicated(subset=[date, ticker], keep=False)
    if duplicated.any():
        first = raw.loc[duplicated].iloc[0]
        raise DataValidationError(
            f"duplicate rows for ticker {first[ticker]!r} on {first[date]!r}; "
            "expected one close per date and ticker"
        )
    # Missing (date, ticker) combinations become NaN here and are reported by
    # validate_prices with the first offending ticker and date.
    # pivot (not pivot_table) because duplicates were rejected above: there is
    # nothing to aggregate.
    return raw.pivot(index=date, columns=ticker, values=close)  # noqa: PD010
