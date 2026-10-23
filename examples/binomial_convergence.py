"""Binomial (CRR) prices converge to the Black-Scholes-Merton price as steps increase.

Run from the repository root:

    uv run python examples/binomial_convergence.py

Prints a few tree prices and writes docs/images/binomial_convergence.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window

import matplotlib.pyplot as plt
import numpy as np

from quantkit.pricing import OptionSpec, binomial_price, bsm_price

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "binomial_convergence.png"


def main() -> None:
    """Price Hull's example call on trees of 1 to 300 steps and plot the error."""
    call = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")
    exact = bsm_price(call)
    steps = np.arange(1, 301)
    tree = np.array([binomial_price(call, steps=int(n)) for n in steps])
    error = np.abs(tree - exact)

    print(f"Black-Scholes-Merton price: {exact:.4f}")
    for n in (10, 50, 100, 300):
        print(f"CRR tree, {n:>3} steps:     {tree[n - 1]:.4f}  (error {error[n - 1]:.1e})")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(steps[4:], tree[4:], lw=0.8, label="CRR binomial tree")
    left.axhline(exact, color="black", ls="--", lw=1, label=f"Black-Scholes-Merton ({exact:.4f})")
    left.set(xlabel="time steps", ylabel="call price", title="Tree price vs. number of steps")
    left.legend()

    right.loglog(steps, error, lw=0.8, label="|tree - BSM|")
    right.loglog(steps, 0.5 / steps, color="black", ls="--", lw=1, label="0.5 / steps")
    right.set(xlabel="time steps", ylabel="absolute error", title="Error shrinks like 1 / steps")
    right.legend()

    fig.suptitle("European call from Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 years")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110)
    print(f"saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
