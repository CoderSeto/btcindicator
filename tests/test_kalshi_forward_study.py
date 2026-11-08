"""
kalshi_forward_study on hand-built recorder output with exact expected answers.

Each hypothesis's rule is driven with a tick frame built by `ticks()` and
checked for: fires on exactly the pre-registered condition, first firing
only, window and mid-gate edges, a missing feed disqualifying the tick (blank,
not zero), and the fill arithmetic (YES at the ask, NO at 1 - bid, fee
rounded up per order). The pre-registration cutoff is tested to refuse rows.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import edge_recorder
import kalshi
import kalshi_forward_study as fs
from edge_recorder import RECORD_FIELDS

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)      # after CONFIRMATORY_FROM
STRIKE = 100000.0


def tick(ts, ticker="M1", secs=200.0, bid=0.50, ask=0.52, bid_sz=10, ask_sz=10, k_up=1, cb=100000.0,
         cb_source="ticker", cb_up=1, p_settle=None, vol=0.5, **extra):
    row = {k: "" for k in RECORD_FIELDS}
    close = ts + timedelta(seconds=secs)
    row.update({"row_kind": "tick", "ts_utc": ts.isoformat(timespec="milliseconds"), "ticker": ticker,
                "strike": STRIKE, "close_time": close.isoformat(), "secs_to_close": secs,
                "k_yes_bid": bid, "k_yes_ask": ask, "k_yes_bid_sz": bid_sz, "k_yes_ask_sz": ask_sz, "k_up": k_up,
                "cb_price": cb, "cb_source": cb_source, "cb_up": cb_up, "vol_ann": vol,
                "model_version": edge_recorder.MODEL_VERSION})
    modelable = cb not in (None, "") and vol not in (None, "") and secs > 0
    if p_settle is None and modelable:
        p_settle = kalshi.settlement_probability(cb, STRIKE, secs / 60, vol)
    row["p_settle"] = "" if p_settle in (None, "") else p_settle
    row["p_point"] = "" if not modelable else kalshi.crossing_probability(cb, STRIKE, secs / 60, vol)
    row.update(extra)
    return row


def gap(ts, seconds):
    row = {k: "" for k in RECORD_FIELDS}
    row.update({"row_kind": "gap", "ts_utc": ts.isoformat(timespec="milliseconds"), "gap_seconds": seconds})
    return row


def record(rows):
    df = pd.DataFrame(rows, columns=RECORD_FIELDS).astype(str).replace("nan", "")
    return fs.load_record(df)


def trades(hyp, rows):
    return fs.trades_for(hyp, record(rows))


# --- loading and cutoff --------------------------------------------------------

def test_rows_before_the_preregistration_are_refused():
    early = T0 - timedelta(days=2)
    assert early < fs.CONFIRMATORY_FROM
    rec = record([tick(early), tick(T0), gap(T0 + timedelta(seconds=10), 30.0)])
    assert rec["refused"] == 1 and len(rec["ticks"]) == 1 and len(rec["gaps"]) == 1
    assert np.isnan(rec["ticks"]["book_imb"].iloc[0])          # blank stays missing, never zero


# --- fills ---------------------------------------------------------------------

def test_fill_prices_and_fee():
    r = record([tick(T0, bid=0.48, ask=0.53, bid_sz=1, ask_sz=1)])["ticks"].iloc[0]
    yes, no = fs.fill("yes", r), fs.fill("no", r)
    assert yes == {"price": 0.53, "fee": 0.02}                 # 0.07*0.53*0.47 = 1.74c -> 2c
    assert no["price"] == pytest.approx(0.52) and no["fee"] == 0.02


def test_fill_needs_a_real_book_and_size_at_the_touch():
    assert fs.fill("yes", record([tick(T0, ask_sz=0)])["ticks"].iloc[0]) is None
    assert fs.fill("no", record([tick(T0, bid_sz="")])["ticks"].iloc[0]) is None
    assert fs.fill("yes", record([tick(T0, k_up=0)])["ticks"].iloc[0]) is None
    assert fs.fill("yes", record([tick(T0, bid=0.001, ask=0.55)])["ticks"].iloc[0]) is None   # 55c spread
    assert fs.fill("yes", record([tick(T0, bid=0.30, ask=0.50)])["ticks"].iloc[0]) is not None  # exactly 20c


# --- H1 --------------------------------------------------------------------------

def h1_rows(move_bp=8.0, quote_changes=False, with_gap=False, next_dt=1.0, cb_source="ticker"):
    rows = []
    t = T0
    for i in range(12):                              # 12 s of a flat quote and a flat spot
        rows.append(tick(t + timedelta(seconds=i), secs=200.0 - i, bid=0.50, ask=0.52, cb=100000.0))
    if with_gap:
        rows.append(gap(t + timedelta(seconds=11.5), 6.0))
    spot = 100000.0 * (1 + move_bp / 1e4)
    bid, ask = (0.51, 0.53) if quote_changes else (0.50, 0.52)
    fire = tick(t + timedelta(seconds=12), secs=188.0, bid=bid, ask=ask, cb=spot, cb_source=cb_source)
    rows.append(fire)
    rows.append(tick(t + timedelta(seconds=12 + next_dt), secs=188.0 - next_dt, bid=0.50, ask=0.53, cb=spot))
    return rows


def test_h1_fires_on_a_stale_quote_after_a_move_and_fills_on_the_next_tick():
    t = trades("H1", h1_rows())
    assert len(t) == 1
    tr = t.iloc[0]
    assert tr["side"] == "yes"
    assert tr["price"] == 0.53                    # the NEXT tick's ask, not the 0.52 standing when it fired
    assert tr["ts"] == pd.Timestamp(T0 + timedelta(seconds=13))
    assert tr["fee"] == 0.02


def test_h1_does_not_fire_when_the_quote_moved_too_small_a_move_or_a_gap_or_candle_spot():
    assert trades("H1", h1_rows(quote_changes=True)).empty
    assert trades("H1", h1_rows(move_bp=4.0)).empty
    assert trades("H1", h1_rows(with_gap=True)).empty
    assert trades("H1", h1_rows(cb_source="candles")).empty
    assert trades("H1", h1_rows(next_dt=4.0)).empty              # next tick too late to charge latency honestly


def test_h1_down_move_buys_no_and_needs_the_model_to_clear_the_price():
    t = trades("H1", h1_rows(move_bp=-8.0))
    assert len(t) == 1 and t.iloc[0]["side"] == "no" and t.iloc[0]["price"] == pytest.approx(0.50)
    # With spot 8bp above the strike at 188s, p_settle ~ 0.95; a 0.52 ask clears easily. Put the ask
    # where the margin fails and nothing trades.
    rows = h1_rows()
    for r in rows:
        r["k_yes_bid"], r["k_yes_ask"] = 0.96, 0.98
    assert trades("H1", rows).empty


# --- H2 ----------------------------------------------------------------------------

def test_h2_trades_toward_the_model_only_inside_60_to_180s_with_a_2c_margin():
    # spot 6bp over the strike: p_settle = 0.774 at 120s (vol 0.5); a 0.70 ask clears by > 2c.
    rows = [tick(T0, secs=200.0, bid=0.68, ask=0.70, cb=100060.0),          # outside the window
            tick(T0 + timedelta(seconds=80), secs=120.0, bid=0.68, ask=0.70, cb=100060.0),
            tick(T0 + timedelta(seconds=81), secs=119.0, bid=0.68, ask=0.70, cb=100060.0)]
    t = trades("H2", rows)
    assert len(t) == 1 and t.iloc[0]["side"] == "yes" and t.iloc[0]["secs_to_close"] == 120.0
    p = kalshi.settlement_probability(100060.0, STRIKE, 2.0, 0.5)
    assert p == pytest.approx(0.7736, abs=1e-4) and p - 0.70 - 0.02 >= fs.H2_MARGIN
    # Push the ask to within 2c of the model: no trade.
    tight = [tick(T0 + timedelta(seconds=80), secs=120.0, bid=p - 0.03, ask=p - 0.01, cb=100060.0)]
    assert trades("H2", tight).empty
    assert trades("H2", [tick(T0, secs=59.0, bid=0.68, ask=0.70, cb=100060.0)]).empty
    assert trades("H2", [tick(T0, secs=120.0, bid=0.68, ask=0.70, cb=100060.0, vol="", p_settle="")]).empty


def test_h2_buys_no_when_spot_is_below_the_strike():
    t = trades("H2", [tick(T0, secs=120.0, bid=0.30, ask=0.32, cb=99940.0)])   # p_settle 0.226
    assert len(t) == 1 and t.iloc[0]["side"] == "no" and t.iloc[0]["price"] == pytest.approx(0.70)


# --- H3, H4, H5 ----------------------------------------------------------------------

BOOK = dict(book_up=1, book_venues_up=2, book_imb=0.25, book_persist_s=45.0, book_persist_dir="bullish")


def test_h3_fires_once_per_market_on_a_persistent_lean_inside_the_mid_gate():
    rows = [tick(T0, secs=250.0, **BOOK), tick(T0 + timedelta(seconds=1), secs=249.0, **BOOK),
            tick(T0 + timedelta(seconds=2), ticker="M2", secs=249.0, **dict(BOOK, book_persist_dir="bearish", book_imb=-0.3))]
    t = trades("H3", rows)
    assert list(t["ticker"]) == ["M1", "M2"] and list(t["side"]) == ["yes", "no"]
    assert t.iloc[0]["ts"] == pd.Timestamp(T0)


@pytest.mark.parametrize("change", [
    dict(book_persist_s=29.0), dict(book_imb=0.17), dict(book_venues_up=1), dict(book_up=0),
    dict(book_imb=""), dict(bid=0.62, ask=0.64), dict(secs=301.0), dict(secs=59.0), dict(cb_up=0, cb=""),
])
def test_h3_edge_cases_do_not_fire(change):
    kw = dict(BOOK)
    kw.update(change)
    assert trades("H3", [tick(T0, **kw)]).empty


def test_h3_mid_gate_boundaries_inclusive():
    assert len(trades("H3", [tick(T0, bid=0.39, ask=0.41, **BOOK)])) == 1      # mid 0.40
    assert len(trades("H3", [tick(T0, bid=0.59, ask=0.61, **BOOK)])) == 1      # mid 0.60
    assert trades("H3", [tick(T0, bid=0.60, ask=0.62, **BOOK)]).empty          # mid 0.61


def test_h4_threshold_is_plus_minus_030_and_needs_two_venues():
    agg = dict(agg_up=1, agg_venues_up=3)
    assert trades("H4", [tick(T0, agg_delta=0.30, **agg)]).iloc[0]["side"] == "yes"
    assert trades("H4", [tick(T0, agg_delta=-0.30, **agg)]).iloc[0]["side"] == "no"
    assert trades("H4", [tick(T0, agg_delta=0.29, **agg)]).empty
    assert trades("H4", [tick(T0, agg_delta=0.9, agg_up=1, agg_venues_up=1)]).empty
    assert trades("H4", [tick(T0, agg_delta="", agg_up=0, agg_venues_up=0)]).empty


def test_h5_longs_liquidated_buys_no_shorts_buys_yes_both_means_nothing():
    assert trades("H5", [tick(T0, liq_up=1, liq_long_notional=1e6, liq_short_notional=0)]).iloc[0]["side"] == "no"
    assert trades("H5", [tick(T0, liq_up=1, liq_long_notional=0, liq_short_notional=2e6)]).iloc[0]["side"] == "yes"
    assert trades("H5", [tick(T0, liq_up=1, liq_long_notional=1e6, liq_short_notional=1e6)]).empty
    assert trades("H5", [tick(T0, liq_up=1, liq_long_notional=999_999.0, liq_short_notional=0)]).empty
    assert trades("H5", [tick(T0, liq_up=0, liq_long_notional="", liq_short_notional="")]).empty


def test_rule_fires_but_nothing_at_the_touch_keeps_looking_in_the_same_market():
    rows = [tick(T0, secs=250.0, ask_sz=0, **BOOK), tick(T0 + timedelta(seconds=1), secs=249.0, ask=0.53, **BOOK)]
    t = trades("H3", rows)
    assert len(t) == 1 and t.iloc[0]["price"] == 0.53


# --- settlement, statistics, verdicts -------------------------------------------------

def test_settle_pnl_exact_and_pending_kept_apart():
    tr = pd.DataFrame([{"hyp": "H3", "ticker": "A", "ts": pd.Timestamp(T0), "side": "yes", "price": 0.52, "fee": 0.02,
                        "secs_to_close": 200.0, "vol_ann": 0.5, "day": "2026-11-09", "hour_bin": "12-18", "p_settle": 0.6, "mid": 0.51, "bid": 0.50, "ask": 0.52},
                       {"hyp": "H3", "ticker": "B", "ts": pd.Timestamp(T0), "side": "no", "price": 0.50, "fee": 0.02,
                        "secs_to_close": 200.0, "vol_ann": 0.5, "day": "2026-11-09", "hour_bin": "12-18", "p_settle": 0.4, "mid": 0.51, "bid": 0.50, "ask": 0.52},
                       {"hyp": "H3", "ticker": "C", "ts": pd.Timestamp(T0), "side": "yes", "price": 0.52, "fee": 0.02,
                        "secs_to_close": 200.0, "vol_ann": 0.5, "day": "2026-11-09", "hour_bin": "12-18", "p_settle": 0.6, "mid": 0.51, "bid": 0.50, "ask": 0.52}])
    s = fs.settle(tr, {"A": "yes", "B": "yes"})
    assert list(s["pending"]["ticker"]) == ["C"]
    got = dict(zip(s["settled"]["ticker"], s["settled"]["pnl"]))
    assert got["A"] == pytest.approx(1 - 0.52 - 0.02) and got["B"] == pytest.approx(0 - 0.50 - 0.02)
    worse = dict(zip(s["settled"]["ticker"], s["settled"]["pnl_worse_1c"]))
    assert worse["A"] == pytest.approx(1 - 0.53 - 0.02)


def test_verdict_order_inconclusive_first():
    assert fs.verdict({"n": 50, "mean": 0.05, "se": 0.01}) == "inconclusive"             # half-width 2.58c
    assert fs.verdict({"n": 50, "mean": 0.03, "se": 0.005}) == "passes confirmation"
    assert fs.verdict({"n": 50, "mean": -0.03, "se": 0.005}) == "no edge found (rejected)"
    assert fs.verdict({"n": 50, "mean": 0.005, "se": 0.005}) == "no edge found (not detected)"
    assert fs.verdict({"n": 0, "mean": np.nan, "se": np.nan}).startswith("inconclusive")


def test_stop_rule_by_n_eff_or_date():
    assert not fs.stop_reached({"n_eff": 100.0}, T0)
    assert fs.stop_reached({"n_eff": 2630.0}, T0)
    assert fs.stop_reached({"n_eff": 1.0}, fs.STOP_DATE)


def test_permutation_null_flips_sides_not_outcomes():
    # 40 YES buys at 0.50/0.52 that all won: a coin flip on side would have lost half of them.
    won, side = np.ones(40), np.array(["yes"] * 40)
    bid, ask = np.full(40, 0.50), np.full(40, 0.52)
    assert fs.permutation_p(won, side, bid, ask, resamples=400) < 0.01
    # Half won, half lost, symmetric book: direction carried no information, p is large.
    won = np.array([1.0] * 20 + [0.0] * 20)
    assert fs.permutation_p(won, side, bid, ask, resamples=400) > 0.3
    # A flipped trade pays the OTHER side's price: asymmetric book, all flips -> exact mean.
    won, side, bid, ask = np.array([1.0]), np.array(["yes"]), np.array([0.40]), np.array([0.45])
    p = fs.permutation_p(won, side, bid, ask, resamples=50)
    assert 0.0 < p < 1.0            # the only alternative (NO at 0.60, lost) is worse: p = share of no-flip draws
    assert fs.permutation_p(np.array([]), np.array([]), np.array([]), np.array([])) != fs.permutation_p(won, side, bid, ask)


def test_concentration_and_vol_terciles():
    t = pd.DataFrame({"ticker": list("ABCDEFG"), "pnl": [0.5, 0.4, 0.3, 0.2, 0.1, -0.4, -0.5]})
    c = fs.concentration(t, top=5)
    assert c["top_share"] == pytest.approx(1.5 / 0.6) and c["mean_without_top"] == pytest.approx(-0.45)
    edges = fs.vol_tercile_edges(pd.DataFrame({"vol_ann": np.linspace(0.1, 1.0, 10)}))
    assert edges[0] < edges[1]


def test_model_drift_is_zero_on_consistent_rows_and_counts_other_versions():
    rec = record([tick(T0), tick(T0 + timedelta(seconds=1), model_version="other/1")])
    d = fs.model_drift(rec["ticks"], edge_recorder.MODEL_VERSION)
    assert d["other_version_rows"] == 1 and d["checked"] == 2 and d["max_abs_gap"] < 1e-9


def test_evaluate_end_to_end_reports_every_hypothesis_and_labels_interim():
    rows = []
    for k in range(6):
        ts = T0 + timedelta(minutes=15 * k)
        rows.append(tick(ts, ticker=f"M{k}", secs=250.0, cb=100060.0 if k % 2 else 99940.0,
                         bid=0.50, ask=0.52, agg_up=1, agg_venues_up=2, agg_delta=0.5 if k % 2 else -0.5, **BOOK))
        rows.append(tick(ts + timedelta(seconds=130), ticker=f"M{k}", secs=120.0, bid=0.50, ask=0.52,
                         cb=100060.0 if k % 2 else 99940.0))
    outcomes = {f"M{k}": ("yes" if k % 2 else "no") for k in range(5)}      # M5 pending
    ev = fs.evaluate(record(rows), outcomes, T0 + timedelta(days=1), edge_recorder.MODEL_VERSION)
    assert set(ev["results"]) == set(fs.HYPOTHESES)
    h4 = ev["results"]["H4"]
    assert h4["stats"]["n"] == 5 and h4["pending"] == 1 and not h4["stop_reached"]
    assert h4["sides"] == {"yes": 2, "no": 3} or h4["sides"] == {"no": 3, "yes": 2}
    assert ev["results"]["H3"]["stats"]["n"] == 5 and ev["results"]["H2"]["stats"]["n"] == 5
    assert ev["results"]["H1"]["stats"]["n"] == 0 and ev["results"]["H5"]["stats"]["n"] == 0
    text = fs.report(ev)
    assert "INTERIM, NOT A VERDICT" in text and "H5:" in text and "Overlap" in text
    assert "upper bound" in text
    assert ev["h2_scoring"]["n"] == 5
    ov = ev["overlap"]
    row = ov[(ov["a"] == "H3") & (ov["b"] == "H4")].iloc[0]
    assert row["same_side_markets"] == 2          # H3 is always bullish/yes here, H4 yes only on odd k


def test_runner_refuses_without_a_record(tmp_path, monkeypatch):
    import importlib
    import sys
    sys.path.insert(0, "review")
    runner = importlib.import_module("kalshi_forward_edge")
    with pytest.raises(SystemExit):
        runner.main(["--record", str(tmp_path / "none.csv"), "--offline"])


def test_runner_offline_prints_report(tmp_path, monkeypatch, capsys):
    import importlib
    import sys
    sys.path.insert(0, "review")
    runner = importlib.import_module("kalshi_forward_edge")
    path = tmp_path / "edge_record.csv"
    pd.DataFrame([tick(T0, **BOOK)], columns=RECORD_FIELDS).to_csv(path, index=False)
    monkeypatch.setattr(runner.config, "KALSHI_OUTCOMES_CACHE_PATH", str(tmp_path / "cache.json"))
    runner.main(["--record", str(path), "--offline"])
    out = capsys.readouterr().out
    assert "H3:" in out and "pending = 1" in out
