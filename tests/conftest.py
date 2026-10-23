"""Shared test configuration and fixtures."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

# Pricing calls are fast, but the first call into SciPy can be slow enough to trip
# Hypothesis' per-example deadline on a cold CI runner.
settings.register_profile("quantkit", deadline=None, max_examples=200)
settings.load_profile("quantkit")


@pytest.fixture
def make_cov() -> Callable[..., pd.DataFrame]:
    """Factory for random, well-conditioned annualized covariance matrices."""

    def factory(n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        vols = rng.uniform(0.08, 0.40, n_assets)
        # A sample correlation of 4n Gaussian draws is positive definite and has
        # realistic, moderately dispersed correlations.
        corr = np.corrcoef(rng.standard_normal((n_assets, 4 * n_assets)))
        labels = [f"A{i}" for i in range(n_assets)]
        return pd.DataFrame(np.outer(vols, vols) * corr, index=labels, columns=labels)

    return factory
