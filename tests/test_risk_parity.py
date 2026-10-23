"""Risk parity: equal contributions, closed-form special cases and known bounds."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quantkit.exceptions import ConvergenceError, InvalidParameterError
from quantkit.portfolio import min_variance, portfolio_volatility, risk_contributions, risk_parity

CovFactory = Callable[..., pd.DataFrame]


@pytest.mark.parametrize("n_assets", [2, 5, 12])
@pytest.mark.parametrize("seed", range(4))
def test_risk_contributions_are_equal(make_cov: CovFactory, n_assets: int, seed: int) -> None:
    cov = make_cov(n_assets, seed)
    weights = risk_parity(cov)
    contributions = risk_contributions(weights, cov)
    np.testing.assert_allclose(contributions, 1.0 / n_assets, atol=1e-9)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_uncorrelated_assets_get_inverse_volatility_weights() -> None:
    vols = np.array([0.1, 0.2, 0.4])
    cov = pd.DataFrame(np.diag(vols**2), index=list("XYZ"), columns=list("XYZ"))
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(risk_parity(cov), expected, rtol=1e-9)


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.3, 0.95])
def test_two_assets_get_inverse_volatility_weights_for_any_correlation(rho: float) -> None:
    # With two assets, equal contributions reduce to w1 * sigma1 = w2 * sigma2.
    s1, s2 = 0.15, 0.35
    cov = pd.DataFrame(
        [[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]], index=["X", "Y"], columns=["X", "Y"]
    )
    weights = risk_parity(cov)
    assert weights["X"] == pytest.approx(s2 / (s1 + s2), rel=1e-9)
    assert weights["Y"] == pytest.approx(s1 / (s1 + s2), rel=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_volatility_lies_between_min_variance_and_equal_weight(
    make_cov: CovFactory, seed: int
) -> None:
    # Maillard, Roncalli & Teiletche (2010): sigma_MV <= sigma_ERC <= sigma_1/n.
    cov = make_cov(8, seed)
    equal = pd.Series(1 / 8, index=cov.index)
    erc_vol = portfolio_volatility(risk_parity(cov), cov)
    assert portfolio_volatility(min_variance(cov), cov) <= erc_vol + 1e-12
    assert erc_vol <= portfolio_volatility(equal, cov) + 1e-12


def test_custom_risk_budgets_are_matched(make_cov: CovFactory) -> None:
    cov = make_cov(4, 7)
    budgets = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    weights = risk_parity(cov, budgets=budgets)
    np.testing.assert_allclose(risk_contributions(weights, cov), budgets, atol=1e-9)


def test_risk_contributions_by_hand() -> None:
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["X", "Y"], columns=["X", "Y"])
    weights = pd.Series({"X": 0.5, "Y": 0.5})
    # Variance 0.25 * 0.04 + 0.25 * 0.01 = 0.0125, of which X contributes 0.01.
    contributions = risk_contributions(weights, cov)
    assert contributions["X"] == pytest.approx(0.8)
    assert contributions["Y"] == pytest.approx(0.2)


def test_rejects_invalid_budgets(make_cov: CovFactory) -> None:
    cov = make_cov(3)
    with pytest.raises(InvalidParameterError, match="sum to one"):
        risk_parity(cov, budgets=pd.Series([0.5, 0.3, 0.3], index=cov.index))
    with pytest.raises(InvalidParameterError, match="positive"):
        risk_parity(cov, budgets=pd.Series([1.2, -0.1, -0.1], index=cov.index))
    with pytest.raises(InvalidParameterError, match="covering"):
        risk_parity(cov, budgets=pd.Series([0.5, 0.5], index=cov.index[:2]))


def test_rejects_invalid_contribution_inputs(make_cov: CovFactory) -> None:
    cov = make_cov(3)
    with pytest.raises(InvalidParameterError, match="covering"):
        risk_contributions(pd.Series([1.0], index=cov.index[:1]), cov)
    with pytest.raises(InvalidParameterError, match="variance"):
        risk_contributions(pd.Series(0.0, index=cov.index), cov)


def test_reports_non_convergence(make_cov: CovFactory) -> None:
    with pytest.raises(ConvergenceError, match="sweeps"):
        risk_parity(make_cov(6), tol=1e-15, max_sweeps=1)
