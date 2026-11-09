import pandas as pd
import pytest

import edge_hunt


def _frame(days):
    return pd.DataFrame({"day": days, "x": range(len(days))})


def test_load_exploration_keeps_only_days_before_the_cutoff(tmp_path):
    p = tmp_path / "obs.csv"
    _frame(["2026-08-17", "2026-08-18", "2026-01-01", "2026-11-08"]).to_csv(p, index=False)
    got = edge_hunt.load_exploration(str(p))
    assert sorted(got["day"]) == ["2026-01-01", "2026-08-17"]
    held = edge_hunt.load_holdout(str(p))
    assert sorted(held["day"]) == ["2026-08-18", "2026-11-08"]


def test_assert_exploration_raises_on_a_sealed_row():
    with pytest.raises(edge_hunt.HoldoutViolation):
        edge_hunt.assert_exploration(_frame(["2026-08-17", "2026-08-18"]))
    edge_hunt.assert_exploration(_frame(["2026-08-17"]))


def test_cutoff_is_the_last_quarter_of_days():
    assert edge_hunt.HOLDOUT_CUTOFF_DAY == "2026-08-18"
