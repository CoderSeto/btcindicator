"""
Tests for kalshi_tracking.py: prediction logging, outcome resolution and the
calibration math. Calibration expectations are worked out by hand in the
comments; kalshi_api.get_market is stubbed (no network).
"""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import kalshi_api
import kalshi_tracking as kt

COLUMNS = ["timestamp", "ticker", "strike", "current_price", "minutes_remaining",
           "p_yes", "annual_vol", "kalshi_yes_bid", "kalshi_yes_ask", "kalshi_last_price",
           "p_yes_point"]


def _cfg(tmp_path, enabled=True):
    return SimpleNamespace(
        LOG_KALSHI_PREDICTIONS=enabled,
        KALSHI_PREDICTIONS_LOG_PATH=str(tmp_path / "preds.csv"),
        KALSHI_OUTCOMES_CACHE_PATH=str(tmp_path / "outcomes.json"),
        KALSHI_CALIBRATION_BUCKET_SIZE=0.2,
        KALSHI_CALIBRATION_MIN_SAMPLE=2,
    )


# ---------- log_prediction ----------

def test_log_disabled_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)
    kt.log_prediction(cfg, "T1", 100, 101, 5, 0.6, 0.5)
    assert not (tmp_path / "preds.csv").exists()


def test_log_writes_header_once_and_fixed_width_rows(tmp_path):
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, "T1", 100, 101, 5, 0.6, 0.5)                                    # no quotes at all
    kt.log_prediction(cfg, "T1", 100, 101, 4, 0.7, 0.5, kalshi_yes_bid=0.6, kalshi_yes_ask=0.7)
    kt.log_prediction(cfg, "T1", 100, 101, 3, 0.8, 0.5, 0.7, 0.8, 0.75)                    # all quotes
    with open(cfg.KALSHI_PREDICTIONS_LOG_PATH, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == COLUMNS
    assert sum(1 for r in rows if r[0] == "timestamp") == 1
    assert {len(r) for r in rows} == {len(COLUMNS)}          # no row wider/narrower than header


def test_log_missing_values_are_blank_not_zero(tmp_path):
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, "T1", 100, 101, 5, 0.6, 0.5)
    kt.log_prediction(cfg, "T1", 100, 101, 4, 0.6, 0.5, kalshi_yes_bid=0.0, kalshi_yes_ask=0.0)
    with open(cfg.KALSHI_PREDICTIONS_LOG_PATH, newline="") as f:
        first, second = list(csv.DictReader(f))
    assert first["kalshi_yes_bid"] == "" and first["kalshi_last_price"] == ""
    assert second["kalshi_yes_bid"] == "0.0"                 # a real zero stays a zero
    assert second["kalshi_last_price"] == ""


def test_log_round_trips_through_pandas(tmp_path):
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, "KXBTC15M-X", 105000.5, 104900.25, 7.5, 0.4321, 0.55)
    kt.log_prediction(cfg, None, 105000.5, 104900.25, 7.0, 0.4, 0.55)
    df = kt.load_predictions(cfg.KALSHI_PREDICTIONS_LOG_PATH)
    assert list(df.columns) == COLUMNS
    assert df.loc[0, "ticker"] == "KXBTC15M-X" and df.loc[0, "p_yes"] == pytest.approx(0.4321)
    assert pd.isna(df.loc[1, "ticker"])                      # no ticker -> blank
    assert df["kalshi_yes_bid"].isna().all()
    pd.to_datetime(df["timestamp"])                          # parseable ISO timestamps


# ---------- outcomes cache ----------

def test_cache_missing_corrupt_and_roundtrip(tmp_path):
    p = str(tmp_path / "c.json")
    assert kt.load_outcomes_cache(p) == {}
    (tmp_path / "c.json").write_text("{not json")
    assert kt.load_outcomes_cache(p) == {}
    kt.save_outcomes_cache(p, {"A": "yes", "B": "no"})
    assert kt.load_outcomes_cache(p) == {"A": "yes", "B": "no"}
    assert json.loads((tmp_path / "c.json").read_text()) == {"A": "yes", "B": "no"}


# ---------- resolve_outcomes ----------

@pytest.fixture
def fake_market(monkeypatch):
    calls = []
    markets = {}

    def get_market(ticker):
        calls.append(ticker)
        m = markets[ticker]
        if isinstance(m, Exception):
            raise m
        return m
    monkeypatch.setattr(kalshi_api, "get_market", get_market)
    return markets, calls


def test_resolve_outcomes_only_caches_settled(fake_market):
    markets, calls = fake_market
    markets.update({
        "Y": {"result": "yes"}, "N": {"result": "no"},
        "OPEN": {"result": ""}, "NONE": {}, "ODD": {"result": "void"},
        "BOOM": RuntimeError("network"),
    })
    cache = {"DONE": "yes"}
    out = kt.resolve_outcomes(["Y", "N", "OPEN", "NONE", "ODD", "BOOM", "DONE", "", None], cache)
    assert out == {"DONE": "yes", "Y": "yes", "N": "no"}
    assert "DONE" not in calls                                # cached -> not re-queried
    assert cache == {"DONE": "yes"}                           # input not mutated
    assert "" not in calls and None not in calls


def test_resolve_outcomes_unresolved_is_rechecked_next_run(fake_market):
    markets, calls = fake_market
    markets["T"] = {"result": ""}
    cache = kt.resolve_outcomes(["T"], {})
    assert cache == {}
    markets["T"] = {"result": "no"}
    assert kt.resolve_outcomes(["T"], cache) == {"T": "no"}
    assert calls == ["T", "T"]


# ---------- compute_calibration ----------

def _df(rows):
    """rows: (ticker, p_yes, minutes_remaining, bid, ask)"""
    return pd.DataFrame(rows, columns=["ticker", "p_yes", "minutes_remaining", "kalshi_yes_bid", "kalshi_yes_ask"])


def test_calibration_hand_computed_tables_and_brier():
    nan = np.nan
    df = _df([
        ("T1", 0.10, 1, 0.05, 0.15),   # no
        ("T2", 0.15, 1, nan, nan),     # yes
        ("T3", 0.50, 1, 0.45, 0.55),   # yes
        ("T4", 0.85, 1, 0.90, 1.00),   # yes
        ("T5", 1.00, 1, nan, nan),     # yes -- p == 1.0 belongs in the last bucket
        ("T6", 0.20, 1, nan, nan),     # no  -- p == 0.2 belongs in 20-40%, not 0-20%
        ("T7", 0.60, 1, nan, nan),     # unresolved -> dropped
    ])
    outcomes = {"T1": "no", "T2": "yes", "T3": "yes", "T4": "yes", "T5": "yes", "T6": "no"}
    res = kt.compute_calibration(df, outcomes, bucket_size=0.2, min_sample=2)

    assert res["n_resolved"] == 6
    m = res["model"].set_index("bucket")
    assert list(m.index) == ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    assert list(m["n"]) == [2, 1, 1, 0, 2]
    assert m.loc["0-20%", "observed_yes_rate"] == 0.5
    assert m.loc["20-40%", "observed_yes_rate"] == 0.0
    assert m.loc["40-60%", "observed_yes_rate"] == 1.0
    assert pd.isna(m.loc["60-80%", "observed_yes_rate"])
    assert m.loc["80-100%", "observed_yes_rate"] == 1.0
    assert list(m["low_sample"]) == [False, True, True, True, False]      # n < 2

    # (0.10^2 + 0.85^2 + 0.5^2 + 0.15^2 + 0 + 0.2^2) / 6
    assert res["model_brier"] == pytest.approx(1.045 / 6)

    # Kalshi mids: T1 0.10 (no), T3 0.50 (yes), T4 0.95 (yes)
    k = res["kalshi_market"].set_index("bucket")
    assert list(k["n"]) == [1, 0, 1, 0, 1]
    assert res["kalshi_brier"] == pytest.approx((0.1 ** 2 + 0.5 ** 2 + 0.05 ** 2) / 3)


def test_calibration_uses_closest_to_close_row_per_ticker():
    df = _df([
        ("A", 0.90, 10, np.nan, np.nan),
        ("A", 0.30, 2, np.nan, np.nan),      # most-informed snapshot
        ("A", 0.60, 6, np.nan, np.nan),
    ])
    res = kt.compute_calibration(df, {"A": "yes"}, 0.2, 1)
    assert res["n_resolved"] == 1               # not pseudo-replicated three times
    assert res["model_brier"] == pytest.approx((0.3 - 1) ** 2)


def test_calibration_kalshi_quote_comes_from_the_same_row_as_the_prediction():
    # The closest-to-close row has NO quote; an older row does. The report must
    # not reach back and borrow the older row's quote for the newer prediction.
    df = _df([
        ("A", 0.90, 10, 0.80, 0.90),
        ("A", 0.30, 2, np.nan, np.nan),
    ])
    res = kt.compute_calibration(df, {"A": "yes"}, 0.2, 1)
    assert res["model_brier"] == pytest.approx(0.49)
    assert res["kalshi_market"] is None
    assert res["kalshi_brier"] is None


def test_calibration_ignores_blank_tickers():
    df = _df([("", 0.5, 1, np.nan, np.nan), ("A", 0.5, 1, np.nan, np.nan)])
    df.loc[0, "ticker"] = ""
    res = kt.compute_calibration(df, {"A": "no", "": "yes"}, 0.2, 1)
    assert res["n_resolved"] == 1


@pytest.mark.parametrize("df,outcomes", [
    (_df([]), {"A": "yes"}),
    (_df([("A", 0.5, 1, np.nan, np.nan)]), {}),
    (_df([("A", 0.5, 1, np.nan, np.nan)]), {"B": "yes"}),    # nothing matches
])
def test_calibration_empty_cases(df, outcomes):
    res = kt.compute_calibration(df, outcomes, 0.2, 5)
    assert res["n_resolved"] == 0
    assert res["model"].empty and res["kalshi_market"] is None
    assert res["model_brier"] is None and res["kalshi_brier"] is None


def test_calibration_perfect_and_coin_flip_brier():
    df = _df([("A", 1.0, 1, np.nan, np.nan), ("B", 0.0, 1, np.nan, np.nan)])
    assert kt.compute_calibration(df, {"A": "yes", "B": "no"}, 0.2, 1)["model_brier"] == 0.0
    df = _df([("A", 0.5, 1, np.nan, np.nan), ("B", 0.5, 1, np.nan, np.nan)])
    assert kt.compute_calibration(df, {"A": "yes", "B": "no"}, 0.2, 1)["model_brier"] == 0.25


def test_calibration_other_bucket_sizes():
    df = _df([("A", 0.1, 1, np.nan, np.nan), ("B", 0.9, 1, np.nan, np.nan)])
    res = kt.compute_calibration(df, {"A": "no", "B": "yes"}, 0.5, 1)
    assert list(res["model"]["bucket"]) == ["0-50%", "50-100%"]
    assert list(res["model"]["n"]) == [1, 1]


# ---------- resolve_and_report (smoke, end to end) ----------

def test_report_without_log_explains_how_to_start(tmp_path, capsys):
    kt.resolve_and_report(_cfg(tmp_path))
    assert "LOG_KALSHI_PREDICTIONS" in capsys.readouterr().out


def test_report_end_to_end(tmp_path, capsys, fake_market):
    markets, _ = fake_market
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, "A", 100, 101, 9, 0.9, 0.5, 0.8, 0.9)
    kt.log_prediction(cfg, "A", 100, 101, 1, 0.7, 0.5, 0.6, 0.7)
    kt.log_prediction(cfg, "B", 100, 99, 1, 0.2, 0.5, 0.2, 0.3)
    markets.update({"A": {"result": "yes"}, "B": {"result": ""}})
    kt.resolve_and_report(cfg)
    out = capsys.readouterr().out
    assert "Loaded 3 logged predictions across 2 unique contracts" in out
    assert "1 contracts resolved so far (out of 2 logged)" in out
    assert "Model Brier score: 0.0900" in out                # (0.7 - 1)^2 on A's closest row
    assert "Kalshi market Brier score: 0.1225" in out        # mid 0.65 -> 0.35^2
    assert json.loads((tmp_path / "outcomes.json").read_text()) == {"A": "yes"}


# ---------- column layout changes between releases (CLAUDE.md gotcha #1) ----------

def test_log_rotates_file_written_under_an_older_layout(tmp_path):
    cfg = _cfg(tmp_path)
    path = tmp_path / "preds.csv"
    old_header = COLUMNS[:-1]                                 # an older release without p_yes_point
    path.write_text(",".join(old_header) + "\nt,T0,100,101,5,0.6,0.5,,\n")

    kt.log_prediction(cfg, "T1", 100, 101, 4, 0.7, 0.5, 0.6, 0.7, 0.65)

    backups = list(tmp_path.glob("preds.csv.*.bak"))
    assert len(backups) == 1
    old = pd.read_csv(backups[0])                             # old data intact and still readable
    assert list(old.columns) == old_header and list(old["ticker"]) == ["T0"]
    new = pd.read_csv(path)                                   # fresh file: current header, one row
    assert list(new.columns) == COLUMNS and list(new["ticker"]) == ["T1"]


def test_log_same_layout_is_appended_not_rotated(tmp_path):
    cfg = _cfg(tmp_path)
    for m in (5, 4, 3):
        kt.log_prediction(cfg, "T1", 100, 101, m, 0.6, 0.5)
    assert not list(tmp_path.glob("*.bak"))
    assert len(pd.read_csv(cfg.KALSHI_PREDICTIONS_LOG_PATH)) == 3


def test_header_constant_matches_what_is_written(tmp_path):
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, "T1", 100, 101, 5, 0.6, 0.5)
    with open(cfg.KALSHI_PREDICTIONS_LOG_PATH, newline="") as f:
        assert next(csv.reader(f)) == kt.PREDICTION_LOG_HEADER == COLUMNS


# ---------- review regressions (see REVIEW_FINDINGS.md) ----------

def test_report_does_not_query_kalshi_for_a_blank_ticker(tmp_path, capsys, fake_market):
    # A blank ticker cell reads back as NaN, and str(NaN) is "nan" -- a truthy string that was
    # counted as a contract and sent to Kalshi as GET /markets/nan on every run.
    markets, calls = fake_market
    cfg = _cfg(tmp_path)
    kt.log_prediction(cfg, None, 100, 101, 5, 0.6, 0.5)
    kt.log_prediction(cfg, "A", 100, 101, 1, 0.7, 0.5, 0.6, 0.7)
    markets["A"] = {"result": "yes"}
    kt.resolve_and_report(cfg)
    assert calls == ["A"]
    assert "across 1 unique contracts" in capsys.readouterr().out


def test_a_prediction_exactly_on_a_bucket_edge_lands_in_the_upper_bucket():
    # Bounds were built as i * 0.2, and 3 * 0.2 == 0.6000000000000001, so a p_yes of exactly 0.6
    # fell into 40-60% instead of 60-80% (lower bounds are inclusive everywhere else).
    df = _df([("A", 0.6, 1.0, None, None), ("B", 0.4, 1.0, None, None)])
    table = kt.compute_calibration(df, {"A": "yes", "B": "no"}, 0.2, 1)["model"].set_index("bucket")
    assert table.loc["60-80%", "n"] == 1
    assert table.loc["40-60%", "n"] == 1



# ---------- fixed-horizon calibration and the point-in-time comparison ----------

def test_calibration_at_a_fixed_horizon_uses_the_row_nearest_it():
    # Closest-to-close rows are nearly decided (p near 0 or 1) and sit where the averaging error
    # is largest, so they measure the model where it matters least. At a 5-minute horizon:
    df = _df([
        ("A", 0.70, 9.0, np.nan, np.nan),
        ("A", 0.60, 5.4, np.nan, np.nan),    # nearest 5 -> used
        ("A", 0.99, 0.2, np.nan, np.nan),    # closest to close -> ignored
        ("B", 0.40, 4.5, np.nan, np.nan),    # used
        ("C", 0.10, 0.5, np.nan, np.nan),    # nothing within +/-1 minute of 5 -> left out
    ])
    res = kt.compute_calibration(df, {"A": "yes", "B": "no", "C": "no"}, 0.2, 1,
                                 horizon_minutes=5.0, horizon_tolerance=1.0)
    assert res["n_resolved"] == 2
    assert res["model_brier"] == pytest.approx(((0.6 - 1) ** 2 + 0.4 ** 2) / 2)


def test_calibration_scores_the_point_in_time_model_alongside_when_logged():
    df = _df([("A", 0.80, 5.0, np.nan, np.nan), ("B", 0.30, 5.0, np.nan, np.nan)])
    df["p_yes_point"] = [0.70, 0.40]
    res = kt.compute_calibration(df, {"A": "yes", "B": "no"}, 0.2, 1, horizon_minutes=5.0, horizon_tolerance=1.0)
    assert res["model_brier"] == pytest.approx((0.2 ** 2 + 0.3 ** 2) / 2)
    assert res["point_brier"] == pytest.approx((0.3 ** 2 + 0.4 ** 2) / 2)


def test_report_states_its_horizon(tmp_path, capsys, fake_market):
    markets, _ = fake_market
    cfg = _cfg(tmp_path)
    cfg.KALSHI_CALIBRATION_HORIZON_MINUTES = 5.0
    cfg.KALSHI_CALIBRATION_HORIZON_TOLERANCE_MINUTES = 1.0
    kt.log_prediction(cfg, "A", 100, 101, 5.2, 0.7, 0.5, 0.6, 0.7, p_yes_point=0.65)
    kt.log_prediction(cfg, "A", 100, 101, 0.1, 0.99, 0.5, 0.98, 0.99, p_yes_point=0.9)
    markets["A"] = {"result": "yes"}
    kt.resolve_and_report(cfg)
    out = capsys.readouterr().out
    assert "5.0 +/- 1.0 minutes before close" in out
    assert "Model Brier score: 0.0900" in out                              # the 5.2-minute row, not 0.1
    assert "Point-in-time model Brier score: 0.1225" in out
