"""Cox-Ross-Rubinstein binomial tree for European and American options."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from quantkit._validation import require_int
from quantkit.exceptions import InvalidParameterError
from quantkit.pricing.option import OptionSpec

ExerciseStyle = Literal["european", "american"]


def binomial_price(
    option: OptionSpec, *, steps: int = 500, exercise: ExerciseStyle = "european"
) -> float:
    """Price an option on a Cox-Ross-Rubinstein (CRR) recombining binomial tree.

    Time to expiry is split into ``steps`` intervals of length ``dt = T / steps``.
    In each interval the underlying moves up by ``u = exp(sigma sqrt(dt))`` or down
    by ``d = 1 / u`` (Cox, Ross & Rubinstein 1979). Matching the risk-neutral
    drift gives the up-move probability::

        p = (exp((r - q) dt) - d) / (u - d)

    Option values are rolled back from expiry, discounting the risk-neutral
    expectation one step at a time. For an American option each node takes the
    larger of that continuation value and the immediate exercise value, which is
    what lets the tree price early exercise.

    The European value converges to Black-Scholes-Merton at rate ``O(1 / steps)``,
    oscillating between odd and even step counts (Hull, chapter "Binomial Trees").

    Args:
        option: Contract and market inputs.
        steps: Number of time steps in the tree (>= 1).
        exercise: ``"european"`` (expiry only) or ``"american"`` (any node).

    Returns:
        The option value at the root of the tree.

    Raises:
        InvalidParameterError: If ``steps`` is invalid, the exercise style is
            unknown, or ``p`` falls outside (0, 1). The latter happens when the
            carry ``|r - q|`` is large relative to ``sigma / sqrt(dt)``; using more
            steps fixes it.
    """
    steps = require_int("steps", steps, minimum=1)
    if exercise not in ("european", "american"):
        raise InvalidParameterError(f"exercise must be 'european' or 'american', got {exercise!r}")

    dt = option.maturity / steps
    log_u = option.vol * math.sqrt(dt)
    up, down = math.exp(log_u), math.exp(-log_u)
    p = (math.exp((option.rate - option.dividend_yield) * dt) - down) / (up - down)
    if not 0.0 < p < 1.0:
        raise InvalidParameterError(
            f"risk-neutral probability {p:.4f} is outside (0, 1); increase steps"
        )
    discount = math.exp(-option.rate * dt)

    # Node j at level i has seen j up-moves and (i - j) down-moves, so its price
    # is S * u**j * d**(i - j) = S * exp(log_u * (2j - i)). Working in logs avoids
    # overflow in u**j for large trees.
    j = np.arange(steps + 1, dtype=np.float64)
    values = option.payoff(option.spot * np.exp(log_u * (2.0 * j - steps)))
    for i in range(steps - 1, -1, -1):
        # values[1:] are the up-children and values[:-1] the down-children of level i.
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            exercise_value = option.payoff(option.spot * np.exp(log_u * (2.0 * j[: i + 1] - i)))
            values = np.maximum(values, exercise_value)
    return float(values[0])
