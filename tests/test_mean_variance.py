"""Mean-variance optimization: SLSQP against closed forms, bounds and random portfolios."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantkit.exceptions import InvalidParameterError
from quantkit.portfolio import (
    efficient_frontier,
    efficient_return,
    efficient_return_closed_form,
    max_sharpe,
    max_sharpe_closed_form,
    min_variance,
    min_variance_closed_form,
    portfolio_performance,
)

CovFactory = Callable[..., pd.DataFrame]
RF = 0.02


def expected_returns_for(cov: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Expected returns loosely increasing with volatility, as in real markets."""
    rng = np.random.default_rng(seed + 1000)
    vols = np.sqrt(np.diag(cov.to_numpy()))
    return pd.Series(0.02 + 0.3 * vols + rng.normal(0.0, 0.02, len(vols)), index=cov.index)


def variance(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.to_numpy()
    return float(w @ cov.to_numpy() @ w)


def sharpe(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> float:
    return portfolio_performance(weights, mu, cov, risk_free_rate=RF).sharpe


def random_long_only(n_assets: int, size: int, seed: int) -> np.ndarray:
    # Dirichlet draws are uniformly distributed on the simplex of long-only weights.
    return np.random.default_rng(seed).dirichlet(np.ones(n_assets), size=size)


# ----------------------------------------------------------------- closed forms


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_min_variance_satisfies_first_order_conditions(
    make_cov: CovFactory, seed: int
) -> None:
    cov = make_cov(6, seed)
    w = min_variance_closed_form(cov)
    # Lagrange condition: Sigma w is the same for every asset.
    marginal = cov.to_numpy() @ w.to_numpy()
    np.testing.assert_allclose(marginal, marginal.mean(), rtol=1e-10)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_tangency_satisfies_first_order_conditions(
    make_cov: CovFactory, seed: int
) -> None:
    cov = make_cov(6, seed)
    mu = expected_returns_for(cov, seed)
    w = max_sharpe_closed_form(mu, cov, risk_free_rate=RF)
    # Sigma w is proportional to the excess returns mu - rf.
    ratio = (cov.to_numpy() @ w.to_numpy()) / (mu.to_numpy() - RF)
    np.testing.assert_allclose(ratio, ratio.mean(), rtol=1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_closed_form_frontier_hits_target_with_hyperbolic_variance(make_cov: CovFactory) -> None:
    cov = make_cov(5, 1)
    mu = expected_returns_for(cov, 1)
    sigma = cov.to_numpy()
    inv = np.linalg.inv(sigma)
    ones = np.ones(len(mu))
    a, b, c = ones @ inv @ ones, ones @ inv @ mu.to_numpy(), mu.to_numpy() @ inv @ mu.to_numpy()
    d = a * c - b * b
    for target in (0.05, 0.08, 0.12, 0.20):
        w = efficient_return_closed_form(mu, cov, target)
        assert w.sum() == pytest.approx(1.0)
        assert float(w.to_numpy() @ mu.to_numpy()) == pytest.approx(target)
        assert variance(w, cov) == pytest.approx((a * target**2 - 2 * b * target + c) / d)


# -------------------------------------------- optimizer vs closed form (no bounds)


@pytest.mark.parametrize("seed", range(5))
def test_optimizer_matches_closed_form_min_variance(make_cov: CovFactory, seed: int) -> None:
    cov = make_cov(6, seed)
    numerical = min_variance(cov, bounds=None)
    exact = min_variance_closed_form(cov)
    np.testing.assert_allclose(numerical, exact, atol=1e-6)
    assert variance(numerical, cov) == pytest.approx(variance(exact, cov), rel=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_optimizer_matches_closed_form_tangency(make_cov: CovFactory, seed: int) -> None:
    cov = make_cov(6, seed)
    mu = expected_returns_for(cov, seed)
    numerical = max_sharpe(mu, cov, risk_free_rate=RF, bounds=None)
    exact = max_sharpe_closed_form(mu, cov, risk_free_rate=RF)
    np.testing.assert_allclose(numerical, exact, atol=1e-6)
    assert sharpe(numerical, mu, cov) == pytest.approx(sharpe(exact, mu, cov), rel=1e-10)


@pytest.mark.parametrize("target", [0.04, 0.08, 0.15])
def test_optimizer_matches_closed_form_efficient_return(
    make_cov: CovFactory, target: float
) -> None:
    cov = make_cov(6, 2)
    mu = expected_returns_for(cov, 2)
    numerical = efficient_return(mu, cov, target, bounds=None)
    exact = efficient_return_closed_form(mu, cov, target)
    np.testing.assert_allclose(numerical, exact, atol=1e-6)
    assert variance(numerical, cov) == pytest.approx(variance(exact, cov), rel=1e-9)


def test_unbounded_frontier_traces_the_closed_form_hyperbola(make_cov: CovFactory) -> None:
    cov = make_cov(5, 3)
    mu = expected_returns_for(cov, 3)
    frontier = efficient_frontier(mu, cov, n_points=8, bounds=None)
    points = zip(frontier["expected_return"], frontier["volatility"], strict=True)
    for target, vol in points:
        exact = efficient_return_closed_form(mu, cov, target)
        assert vol == pytest.approx(np.sqrt(variance(exact, cov)), rel=1e-8)


# ------------------------------------------------------------- bounded problems


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-0.1, 0.4), (0.05, 0.5)])
@pytest.mark.parametrize("seed", range(3))
def test_weights_sum_to_one_and_respect_bounds(
    make_cov: CovFactory, bounds: tuple[float, float], seed: int
) -> None:
    cov = make_cov(6, seed)
    mu = expected_returns_for(cov, seed)
    portfolios = [
        min_variance(cov, bounds=bounds),
        max_sharpe(mu, cov, risk_free_rate=RF, bounds=bounds),
        efficient_return(mu, cov, float(mu.median()), bounds=bounds),
    ]
    for weights in portfolios:
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert weights.min() >= bounds[0]
        assert weights.max() <= bounds[1]
        assert list(weights.index) == list(cov.index)


@pytest.mark.parametrize("seed", range(5))
def test_long_only_min_variance_beats_random_portfolios(make_cov: CovFactory, seed: int) -> None:
    cov = make_cov(6, seed)
    best = variance(min_variance(cov), cov)
    candidates = random_long_only(6, 20_000, seed)
    random_variances = np.einsum("ij,jk,ik->i", candidates, cov.to_numpy(), candidates)
    assert best <= random_variances.min() + 1e-12
    # Constraints can only hurt: the unconstrained optimum is at least as good.
    assert variance(min_variance_closed_form(cov), cov) <= best + 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_tangency_sharpe_beats_random_feasible_portfolios(make_cov: CovFactory, seed: int) -> None:
    cov = make_cov(6, seed)
    mu = expected_returns_for(cov, seed)
    best = sharpe(max_sharpe(mu, cov, risk_free_rate=RF), mu, cov)
    candidates = random_long_only(6, 20_000, seed)
    excess = candidates @ mu.to_numpy() - RF
    vols = np.sqrt(np.einsum("ij,jk,ik->i", candidates, cov.to_numpy(), candidates))
    assert best >= (excess / vols).max() - 1e-9
    # The unconstrained tangency portfolio has the highest Sharpe ratio of all.
    assert sharpe(max_sharpe_closed_form(mu, cov, risk_free_rate=RF), mu, cov) >= best - 1e-9


def test_long_only_frontier_is_monotone_and_starts_at_min_variance(make_cov: CovFactory) -> None:
    cov = make_cov(6, 4)
    mu = expected_returns_for(cov, 4)
    frontier = efficient_frontier(mu, cov, n_points=12, risk_free_rate=RF)
    assert len(frontier) == 12
    assert np.all(np.diff(frontier["expected_return"]) > 0)
    assert np.all(np.diff(frontier["volatility"]) > -1e-9)
    gmv_vol = np.sqrt(variance(min_variance(cov), cov))
    assert frontier["volatility"].iloc[0] == pytest.approx(gmv_vol, rel=1e-6)
    weights = frontier[list(cov.index)]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= 0).all().all()
    tangency = sharpe(max_sharpe(mu, cov, risk_free_rate=RF), mu, cov)
    assert frontier["sharpe"].max() <= tangency + 1e-9


def test_inputs_are_aligned_by_label_not_position(make_cov: CovFactory) -> None:
    cov = make_cov(4, 5)
    mu = expected_returns_for(cov, 5)
    shuffled = mu.iloc[::-1]
    pd.testing.assert_series_equal(
        max_sharpe(mu, cov, risk_free_rate=RF), max_sharpe(shuffled, cov, risk_free_rate=RF)
    )


def test_portfolio_performance_by_hand() -> None:
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["X", "Y"], columns=["X", "Y"])
    mu = pd.Series({"X": 0.06, "Y": 0.10})
    weights = pd.Series({"X": 0.5, "Y": 0.5})
    perf = portfolio_performance(weights, mu, cov, risk_free_rate=0.01)
    expected_vol = np.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.006)
    assert perf.expected_return == pytest.approx(0.08)
    assert perf.volatility == pytest.approx(expected_vol)
    assert perf.sharpe == pytest.approx((0.08 - 0.01) / expected_vol)


# --------------------------------------------------------------------- errors


def test_rejects_infeasible_or_inverted_bounds(make_cov: CovFactory) -> None:
    cov = make_cov(5)
    with pytest.raises(InvalidParameterError, match="infeasible"):
        min_variance(cov, bounds=(0.3, 1.0))  # 5 x 0.3 > 1
    with pytest.raises(InvalidParameterError, match="infeasible"):
        min_variance(cov, bounds=(0.0, 0.1))  # 5 x 0.1 < 1
    with pytest.raises(InvalidParameterError, match="exceeds"):
        min_variance(cov, bounds=(0.5, 0.1))


def test_rejects_unreachable_target_return(make_cov: CovFactory) -> None:
    cov = make_cov(4)
    mu = expected_returns_for(cov)
    with pytest.raises(InvalidParameterError, match="attainable"):
        efficient_return(mu, cov, float(np.max(mu)) + 0.01)


def test_rejects_mismatched_or_invalid_inputs(make_cov: CovFactory) -> None:
    cov = make_cov(4)
    mu = expected_returns_for(cov)
    with pytest.raises(InvalidParameterError, match="same assets"):
        max_sharpe(mu.iloc[:3], cov)
    with pytest.raises(InvalidParameterError, match="Series"):
        max_sharpe(cast("pd.Series", mu.to_numpy()), cov)  # wrong type on purpose
    with pytest.raises(InvalidParameterError, match="NaN"):
        max_sharpe(mu.where(mu > mu.min()), cov)
    not_pd = cov.copy()
    not_pd.iloc[0, 1] = 10.0
    not_pd.iloc[1, 0] = 10.0
    with pytest.raises(InvalidParameterError, match="positive definite"):
        min_variance_closed_form(not_pd)
    values = cov.to_numpy().copy()
    values[0, 1] += 0.01
    asymmetric = pd.DataFrame(values, index=cov.index, columns=cov.columns)
    with pytest.raises(InvalidParameterError, match="symmetric"):
        min_variance(asymmetric)
    with pytest.raises(InvalidParameterError, match="square"):
        min_variance(cov.iloc[:, :3])
    relabelled = cov.copy()
    relabelled.columns = ["W", "X", "Y", "Z"]
    with pytest.raises(InvalidParameterError, match="labels"):
        min_variance(relabelled)


def test_tangency_requires_returns_above_the_risk_free_rate(make_cov: CovFactory) -> None:
    cov = make_cov(4)
    low_mu = pd.Series(0.01, index=cov.index) + np.linspace(0.0, 0.004, 4)
    with pytest.raises(InvalidParameterError, match="risk-free"):
        max_sharpe(low_mu, cov, risk_free_rate=0.05)
    with pytest.raises(InvalidParameterError, match="tangency"):
        max_sharpe_closed_form(low_mu, cov, risk_free_rate=0.05)


def test_degenerate_frontier_inputs_are_rejected(make_cov: CovFactory) -> None:
    cov = make_cov(4)
    flat = pd.Series(0.05, index=cov.index)
    with pytest.raises(InvalidParameterError, match="degenerate"):
        efficient_return_closed_form(flat, cov, 0.05)
    mu = expected_returns_for(cov)
    with pytest.raises(InvalidParameterError, match="must exceed"):
        efficient_frontier(mu, cov, max_return=float(np.min(mu)) - 0.01)
