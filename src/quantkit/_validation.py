"""Input-validation helpers shared by the public API.

Public functions validate their arguments up front and raise a typed error that
names the offending argument, instead of letting a NaN propagate silently or a
cryptic NumPy error surface three calls deeper.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit.exceptions import DataValidationError, InvalidParameterError

FloatArray = npt.NDArray[np.float64]

SeedLike = int | np.random.Generator | None
"""Anything accepted by :func:`numpy.random.default_rng`: a seed, a generator or None."""


def require_finite(name: str, value: float) -> float:
    """Return ``value`` as a float, or raise if it is NaN or infinite."""
    if not math.isfinite(value):
        raise InvalidParameterError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def require_positive(name: str, value: float) -> float:
    """Return ``value`` as a float, or raise unless it is finite and > 0."""
    if not (math.isfinite(value) and value > 0):
        raise InvalidParameterError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def require_int(name: str, value: int, *, minimum: int) -> int:
    """Return ``value``, or raise unless it is an integer >= ``minimum``."""
    # bool is a subclass of int, but `steps=True` is always a bug.
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise InvalidParameterError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidParameterError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def require_level(name: str, value: float) -> float:
    """Return ``value``, or raise unless it lies strictly between 0 and 1."""
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise InvalidParameterError(f"{name} must lie strictly between 0 and 1, got {value!r}")
    return float(value)


def as_1d_array(name: str, values: npt.ArrayLike, *, min_length: int = 1) -> FloatArray:
    """Convert a sequence, array or Series of numbers to a finite 1-D float array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(f"{name} must contain only numbers") from exc
    if array.ndim != 1:
        raise InvalidParameterError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < min_length:
        raise InvalidParameterError(
            f"{name} needs at least {min_length} observations, got {array.size}"
        )
    if not np.isfinite(array).all():
        raise InvalidParameterError(f"{name} must not contain NaN or infinite values")
    return array


def validate_prices(prices: pd.DataFrame, *, require_datetime_index: bool = False) -> pd.DataFrame:
    """Check a wide price table (rows = dates, columns = assets) and return a float copy.

    Args:
        prices: Prices with one row per date and one column per asset.
        require_datetime_index: Also require a strictly increasing DatetimeIndex.

    Returns:
        A float64 copy of ``prices``.

    Raises:
        DataValidationError: If the table is empty, non-numeric, has duplicate
            columns, missing or non-positive values, or badly ordered dates.
    """
    if not isinstance(prices, pd.DataFrame):
        raise DataValidationError(
            f"prices must be a pandas DataFrame, got {type(prices).__name__}"
        )
    if prices.shape[0] < 2 or prices.shape[1] < 1:
        raise DataValidationError(
            "prices need at least two rows (dates) and one column (asset), "
            f"got shape {prices.shape}"
        )
    if prices.columns.has_duplicates:
        duplicated = sorted({str(c) for c in prices.columns[prices.columns.duplicated()]})
        raise DataValidationError(f"duplicate asset columns: {duplicated}")
    if require_datetime_index:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise DataValidationError("prices must be indexed by a pandas DatetimeIndex")
        if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
            raise DataValidationError("price dates must be unique and sorted ascending")
    try:
        values = prices.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("prices must be numeric") from exc
    array = values.to_numpy()
    missing = np.isnan(array)
    if missing.any():
        row, col = np.argwhere(missing)[0]
        raise DataValidationError(
            f"prices contain {int(missing.sum())} missing value(s); the first is "
            f"{values.columns[col]!r} on {values.index[row]}"
        )
    if not np.isfinite(array).all():
        raise DataValidationError("prices must not contain infinite values")
    if (array <= 0).any():
        raise DataValidationError("prices must be strictly positive")
    return values


def validate_returns(returns: pd.DataFrame, *, min_rows: int = 2) -> pd.DataFrame:
    """Check a wide table of periodic returns and return a float copy."""
    if not isinstance(returns, pd.DataFrame):
        raise InvalidParameterError(
            f"returns must be a pandas DataFrame, got {type(returns).__name__}"
        )
    if returns.shape[0] < min_rows or returns.shape[1] < 1:
        raise InvalidParameterError(
            f"returns need at least {min_rows} rows and one column, got shape {returns.shape}"
        )
    try:
        values = returns.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("returns must be numeric") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise InvalidParameterError("returns must not contain NaN or infinite values")
    return values


def validate_covariance(cov: pd.DataFrame) -> FloatArray:
    """Check a labelled covariance matrix and return it as a symmetric float array."""
    if not isinstance(cov, pd.DataFrame):
        raise InvalidParameterError(f"cov must be a pandas DataFrame, got {type(cov).__name__}")
    n_rows, n_cols = cov.shape
    if n_rows == 0 or n_rows != n_cols:
        raise InvalidParameterError(f"cov must be a non-empty square matrix, got {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise InvalidParameterError("cov must have identical row and column labels")
    matrix = cov.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise InvalidParameterError("cov must not contain NaN or infinite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14):
        raise InvalidParameterError("cov must be symmetric")
    # Average with the transpose so round-off asymmetry cannot leak into results.
    symmetric: FloatArray = 0.5 * (matrix + matrix.T)
    return symmetric


def require_positive_definite(matrix: FloatArray, name: str = "cov") -> None:
    """Raise unless ``matrix`` is (numerically) positive definite."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise InvalidParameterError(f"{name} must be positive definite") from exc
