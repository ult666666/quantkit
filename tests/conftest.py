"""Shared test configuration."""

from hypothesis import settings

# Pricing calls are fast, but the first call into SciPy can be slow enough to trip
# Hypothesis' per-example deadline on a cold CI runner.
settings.register_profile("quantkit", deadline=None, max_examples=200)
settings.load_profile("quantkit")
