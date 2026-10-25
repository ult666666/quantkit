# Walk-Forward Risk-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a prior-information-only portfolio construction/backtest workflow runnable and exportable.

**Architecture:** An additive example composes `compute_returns`, `ledoit_wolf_covariance`, `risk_parity`, `rebalance_days`, `backtest_weights` and `performance_summary`. Preserve the library APIs.

**Tech Stack:** Python 3.11+, NumPy/pandas and existing quantkit APIs; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-walk-forward-risk-parity-design.md`

## Global Constraints

- Defaults: 1,260 fictional business days, 252-return window, seed 42, 10 bps costs, initial wealth 1.
- At date t use only closes ending t−1; trade at close t and hold during t+1.
- Shared monthly rebalance calendar, cash warmup and costs for risk parity/equal weight.
- Export prices, both target tables, wealth, summary and estimation-window CSVs.
- No optimization rewrite, live data, performance claim or execution capability.

## Review Focus

- Using date t's close in covariance must fail the current-close perturbation test.
- Insufficient history must produce a clear CLI failure before files are written.
- Both strategies must have identical trade dates and warmup treatment.
- Higher costs with unchanged targets must never improve wealth.
- Exported target blanks mean no trade, and summary excludes the shared cash warmup.

### Task 1: Example and scenario

**Files:** Create `examples/walk_forward_risk_parity.py`, `tests/test_walk_forward_example.py`, `docs/WALK_FORWARD.md`; modify `README.md` and `.gitignore` for the default output directory.

**Interfaces:** `build_targets(prices: pd.DataFrame, lookback: int) -> TargetSchedule` with risk-parity/equal-weight frames and estimation windows; `run_example(output_dir: Path, *, n_days: int, lookback: int, cost_bps: float, seed: int) -> dict[str, Path]`; `main(argv: Sequence[str] | None = None) -> int`.

- [ ] Write meaningful failing tests for prior-information invariance, feasible allocations, shared cash warmup, costs, windows and deterministic actual CLI files.
- [ ] Run `uv run pytest tests/test_walk_forward_example.py -q`; expect failure on the absent example.
- [ ] Implement the minimal typed composition and export workflow; rerun focused tests.
- [ ] Add README scenario, output contract, explicit timing and limits with the measured run dates.
- [ ] Run the full `uv run pytest --cov=quantkit --cov-fail-under=90`, existing ruff/format/mypy checks and the documented workflow. Expect all existing 333 tests plus new tests to pass.
- [ ] Fetch and preserve concurrent main additions, commit, push a scoped feature PR and attach it; parent reviews the exact head, no merge.
