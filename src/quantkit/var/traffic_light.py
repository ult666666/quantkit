"""The Basel Committee's traffic-light classification of VaR backtests.

Banks using internal models compare each day's 99% one-day VaR with the next
day's trading outcome over the most recent 250 trading days, and count the
exceptions. The count puts the model in a zone and sets a "plus factor": an
increase in the multiplier, at least 3, that scales the VaR-based market risk
capital charge (Basel Committee on Banking Supervision 1996, Table 2):

======  ==========  ===========  ======================
zone    exceptions  plus factor  cumulative probability
======  ==========  ===========  ======================
green   0 to 4      0.00         8.11% to 89.22%
yellow  5           0.40         95.88%
yellow  6           0.50         98.63%
yellow  7           0.65         99.60%
yellow  8           0.75         99.89%
yellow  9           0.85         99.97%
red     10 or more  1.00         99.99%
======  ==========  ===========  ======================

The cumulative probability is the chance that a correct model produces that
many exceptions or fewer: a binomial distribution with 250 trials and
probability 1%. The green zone holds the counts whose cumulative probability is
below 95%, and the red zone starts where it reaches 99.99%.

Reference: Basel Committee on Banking Supervision (1996), *Supervisory framework
for the use of "backtesting" in conjunction with the internal models approach to
market risk capital requirements*, https://www.bis.org/publ/bcbs22.htm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quantkit._validation import require_int
from quantkit.exceptions import InvalidParameterError

Zone = Literal["green", "yellow", "red"]
"""A Basel traffic-light zone."""

BASEL_DAYS = 250
"""Number of daily forecasts the traffic light counts exceptions over."""

BASEL_LEVEL = 0.99
"""VaR confidence level the traffic light applies to."""

BASE_MULTIPLIER = 3.0
"""Minimum multiplication factor applied to VaR in the capital charge."""

_YELLOW_PLUS_FACTORS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


@dataclass(frozen=True, slots=True)
class TrafficLight:
    """The Basel zone and capital plus factor for a count of exceptions.

    Attributes:
        zone: ``"green"``, ``"yellow"`` or ``"red"``.
        n_exceptions: Exceptions among the last 250 daily 99% VaR forecasts.
        plus_factor: Increase in the capital multiplier.
    """

    zone: Zone
    n_exceptions: int
    plus_factor: float

    @property
    def multiplier(self) -> float:
        """Capital multiplier: ``3 + plus_factor``."""
        return BASE_MULTIPLIER + self.plus_factor


def basel_traffic_light(n_exceptions: int) -> TrafficLight:
    """Classify the number of exceptions of 250 daily 99% VaR forecasts.

    Args:
        n_exceptions: Exceptions over the most recent 250 trading days. The zones
            are defined for that sample size and for 99% VaR only.

    Returns:
        The zone and plus factor from Table 2 of the Basel Committee's 1996
        backtesting framework.
    """
    count = require_int("n_exceptions", n_exceptions, minimum=0)
    if count > BASEL_DAYS:
        raise InvalidParameterError(
            f"n_exceptions cannot exceed the {BASEL_DAYS} days counted, got {count}"
        )
    if count < 5:
        return TrafficLight("green", count, 0.0)
    if count < 10:
        return TrafficLight("yellow", count, _YELLOW_PLUS_FACTORS[count])
    return TrafficLight("red", count, 1.0)
