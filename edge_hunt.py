"""
Sealed holdout for the KXBTC15M edge hunt (EDGE_RESULTS.md).

The last 25% of days in study_data/model_obs.csv are SEALED: `HOLDOUT_CUTOFF_DAY`
was written down and committed before any modelling on the cache (2026-11-08).
Exploration code loads observations through `load_exploration()`, which raises
on any row stamped on or after the cutoff, so no feature search, grid or fit
can see the holdout. `load_holdout()` is for the single confirmation run a
candidate earns by clearing the pre-cutoff bar; it is never used to tune.

Changing the cutoff after modelling has begun is a labelled amendment in
EDGE_RESULTS.md, never a quiet edit.
"""
import pandas as pd

HOLDOUT_CUTOFF_DAY = "2026-08-18"        # first sealed day (inclusive); 246 of 329 days are exploratory
MODEL_OBS_PATH = "study_data/model_obs.csv"
MODEL_OBS_SHA256_PREFIX = "8e8d50e7dd5f12c0"   # of the file the cutoff was computed on


class HoldoutViolation(RuntimeError):
    """Exploration code touched a sealed row."""


def assert_exploration(df: pd.DataFrame, cutoff: str = HOLDOUT_CUTOFF_DAY) -> pd.DataFrame:
    """Return df unchanged if every row is before the cutoff; raise otherwise."""
    late = df["day"].astype(str) >= cutoff
    if late.any():
        raise HoldoutViolation(f"{int(late.sum())} rows on or after the sealed cutoff {cutoff}")
    return df


def load_exploration(path: str = MODEL_OBS_PATH, cutoff: str = HOLDOUT_CUTOFF_DAY) -> pd.DataFrame:
    """Pre-cutoff observations only (filtered, then asserted, so a later file can't leak)."""
    df = pd.read_csv(path)
    return assert_exploration(df[df["day"].astype(str) < cutoff].reset_index(drop=True), cutoff)


def load_holdout(path: str = MODEL_OBS_PATH, cutoff: str = HOLDOUT_CUTOFF_DAY) -> pd.DataFrame:
    """The sealed rows. ONE evaluation per advancing candidate; never a fit."""
    df = pd.read_csv(path)
    return df[df["day"].astype(str) >= cutoff].reset_index(drop=True)
