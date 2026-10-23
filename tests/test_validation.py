"""Shared input validation: the error messages users see for malformed inputs."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantkit import DataValidationError, InvalidParameterError, QuantkitError
from quantkit._validation import validate_covariance, validate_prices, validate_returns


def test_all_library_errors_share_a_base_class() -> None:
    assert issubclass(DataValidationError, QuantkitError)
    assert issubclass(InvalidParameterError, ValueError)


def test_duplicate_price_columns_are_named() -> None:
    prices = pd.DataFrame([[1.0, 2.0], [1.1, 2.1]], columns=["X", "X"])
    with pytest.raises(DataValidationError, match=r"duplicate asset columns: \['X'\]"):
        validate_prices(prices)


def test_returns_must_be_a_dataframe() -> None:
    with pytest.raises(InvalidParameterError, match="DataFrame"):
        validate_returns(cast("pd.DataFrame", [[0.01], [0.02]]))


def test_covariance_must_be_a_finite_dataframe() -> None:
    with pytest.raises(InvalidParameterError, match="DataFrame"):
        validate_covariance(cast("pd.DataFrame", np.eye(2)))
    cov = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    with pytest.raises(InvalidParameterError, match="NaN"):
        validate_covariance(cov)
