"""
Tests for paper_trading.py: fills, fees, settlement maths, the event ledger and
the account summary. Expected values are worked out by hand in the comments.
kalshi_api.get_market is stubbed (no network). The fee is cross-checked against
kalshi_study's independent per-contract profit functions, which the edge study
(REVIEW_FINDINGS.md) used.
"""
import csv
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

import kalshi_api
import kalshi_study
import paper_trading as pt

NOW = datetime(2026, 11, 7, 18, 7, 30, tzinfo=timezone.utc)
CLOSE = datetime(2026, 11, 7, 18, 15, tzinfo=timezone.utc)


def _cfg(tmp_path):
    return SimpleNamespace(PAPER_TRADE_LOG_PATH=str(tmp_path / "paper.csv"),
                           PAPER_MIN_SAMPLE=30, PAPER_NOISE_Z=2.0, PAPER_AUTO_ENTRY_MINUTES=5.0,
                           PAPER_AUTO_LAST_MINUTES=2.0, PAPER_AUTO_MODEL_MARGIN=0.02)


def market(**over):
    m = {"available": True, "ticker": "KXBTC15M-T1", "strike": 75_000.0, "close_time": CLOSE,
         "yes_bid": 0.43, "yes_ask": 0.44, "last_price": 0.44, "status": "active"}
    m.update(over)
    return m


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch):
    monkeypatch.setattr(pt, "_last_settle_attempt", {})


def rows(cfg):
    with open(cfg.PAPER_TRADE_LOG_PATH, newline="") as f:
        return list(csv.DictReader(f))


def stub_results(monkeypatch, results, calls=None):
    """kalshi_api.get_market answering from {ticker: "yes"/"no"/None}; records each ticker asked."""
    def get_market(ticker):
        if calls is not None:
            calls.append(ticker)
        if ticker not in results:
            raise RuntimeError("Kalshi unreachable")
        return {"ticker": ticker, "result": results[ticker] or ""}
    monkeypatch.setattr(kalshi_api, "get_market", get_market)


# ---------- fills ----------

def test_yes_is_bought_at_the_ask_and_no_at_one_minus_the_bid():
    assert pt.fill_price("yes", 0.43, 0.44) == 0.44
    assert pt.fill_price("no", 0.43, 0.44) == 0.57          # selling YES at 0.43 costs 1 - 0.43


def test_no_fill_price_has_no_float_residue():
    assert pt.fill_price("no", 0.37, 0.4) == 0.63           # 1 - 0.37 is 0.63 in text, not 0.6300000000000001


def test_a_missing_or_degenerate_quote_gives_no_fill():
    assert pt.fill_price("yes", 0.4, None) is None
    assert pt.fill_price("no", None, 0.4) is None
    assert pt.fill_price("yes", 0.0, 0.0) is None            # an empty book quotes 0/0 (see the finalized fixture)
    assert pt.fill_price("yes", 0.99, 1.0) is None
    assert pt.fill_price("no", 1.0, 1.0) is None             # NO would cost nothing


def test_unknown_side_is_an_error_not_a_silent_yes():
    with pytest.raises(ValueError):
        pt.fill_price("maybe", 0.4, 0.5)


# ---------- fees and settlement ----------

def test_fee_is_kalshis_formula_rounded_up_per_order():
    assert pt.order_fee(0.50, 1) == 0.02                     # 0.07 * 0.25 = 0.0175 -> 2c
    assert pt.order_fee(0.50, 100) == 1.75                   # 0.07 * 100 * 0.25
    assert pt.order_fee(0.44, 1) == 0.02                     # 0.07 * 0.2464 = 0.0172
    assert pt.order_fee(0.40, 10) == 0.17                    # 0.07 * 10 * 0.24 = 0.168 -> 17c


def test_order_cost_is_contracts_plus_fee():
    assert pt.order_cost(0.40, 10) == 4.17


def test_winning_and_losing_positions():
    # 10 YES at 40c, fee 17c.  Win: +10.00 - 4.00 - 0.17.  Lose: -4.00 - 0.17.
    assert pt.settlement("yes", 10, 0.40, 0.17, "yes") == {"won": True, "payout": 10.0, "pnl": 5.83}
    assert pt.settlement("yes", 10, 0.40, 0.17, "no") == {"won": False, "payout": 0.0, "pnl": -4.17}
    # NO wins only on a "no" result.
    assert pt.settlement("no", 10, 0.60, 0.17, "no")["pnl"] == 10.0 - 6.0 - 0.17
    assert pt.settlement("no", 10, 0.60, 0.17, "yes")["pnl"] == -6.17


@pytest.mark.parametrize("outcome", [0, 1])
@pytest.mark.parametrize("contracts", [1, 7, 100])
def test_pnl_matches_the_edge_studys_independent_profit_functions(outcome, contracts):
    bid, ask = 0.43, 0.44
    res = "yes" if outcome else "no"
    yes_fee = pt.order_fee(ask, contracts)
    no_price = pt.fill_price("no", bid, ask)
    no_fee = pt.order_fee(no_price, contracts)
    yes_pnl = pt.settlement("yes", contracts, ask, yes_fee, res)["pnl"]
    no_pnl = pt.settlement("no", contracts, no_price, no_fee, res)["pnl"]
    assert yes_pnl == pytest.approx(float(kalshi_study.ev_buy_yes(outcome, ask, contracts)) * contracts, abs=1e-9)
    assert no_pnl == pytest.approx(float(kalshi_study.ev_buy_no(outcome, bid, contracts)) * contracts, abs=1e-9)


# ---------- placing orders ----------

def test_an_order_is_one_fixed_width_open_row(tmp_path):
    cfg = _cfg(tmp_path)
    res = pt.place_order(cfg, market(), "yes", 10, spot=75_100.5, model_p_yes=0.61, composite="bullish", now=NOW)
    assert res["ok"]
    t = res["trade"]
    assert (t["price"], t["fee"], t["cost"]) == (0.44, 0.18, 4.58)     # fee 0.07 * 10 * 0.44 * 0.56 = 0.1725 -> 18c
    r = rows(cfg)
    assert len(r) == 1 and list(r[0]) == pt.LEDGER_HEADER
    o = r[0]
    assert (o["event"], o["ticker"], o["side"], o["contracts"], o["price"]) == ("open", "KXBTC15M-T1", "yes", "10", "0.44")
    assert (o["yes_bid"], o["yes_ask"], o["strike"]) == ("0.43", "0.44", "75000.0")
    assert (o["spot"], o["model_p_yes"], o["composite"]) == ("75100.5", "0.61", "bullish")
    assert o["close_time"] == CLOSE.isoformat()
    assert (o["outcome"], o["payout"], o["pnl"]) == ("", "", "")


def test_optional_context_is_blank_not_zero_when_unknown(tmp_path):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "no", 1, now=NOW)
    o = rows(cfg)[0]
    assert (o["spot"], o["model_p_yes"], o["composite"]) == ("", "", "")
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["model_p_yes"] is None     # not 0.0


def test_no_is_priced_off_the_bid(tmp_path):
    cfg = _cfg(tmp_path)
    t = pt.place_order(cfg, market(yes_bid=0.43, yes_ask=0.44), "no", 1, now=NOW)["trade"]
    assert t["price"] == 0.57


@pytest.mark.parametrize("kwargs, fragment", [
    ({"side": "up"}, "Unknown side"),
    ({"contracts": 0}, "at least 1"),
    ({"contracts": -3}, "at least 1"),
    ({"contracts": 2.5}, "whole contract"),
    ({"mkt": {"available": False, "reason": "No open KXBTC15M market found right now."}}, "No open"),
    ({"mkt": market(close_time=NOW - timedelta(seconds=1))}, "already closed"),
    ({"mkt": market(close_time=NOW)}, "already closed"),
    ({"mkt": market(yes_ask=None)}, "No usable YES quote"),
    ({"mkt": market(yes_bid=None), "side": "no"}, "No usable NO quote"),
])
def test_refused_orders_write_nothing(tmp_path, kwargs, fragment):
    cfg = _cfg(tmp_path)
    mkt = kwargs.get("mkt", market())
    res = pt.place_order(cfg, mkt, kwargs.get("side", "yes"), kwargs.get("contracts", 1), now=NOW)
    assert not res["ok"] and fragment.lower() in res["reason"].lower()
    assert not (tmp_path / "paper.csv").exists()


def test_a_ledger_from_an_older_layout_is_rotated_not_appended_to(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "paper.csv").write_text("event,trade_id\nopen,old\n")
    assert pt.place_order(cfg, market(), "yes", 1, now=NOW)["ok"]
    assert [p.name for p in tmp_path.glob("paper.csv.*.bak")]
    assert len(pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)) == 1


# ---------- the ledger ----------

def test_missing_ledger_is_just_empty(tmp_path):
    assert pt.load_trades(str(tmp_path / "nope.csv")) == []


def test_a_settle_row_closes_its_open_row_and_only_once(tmp_path):
    cfg = _cfg(tmp_path)
    tid = pt.place_order(cfg, market(), "yes", 1, now=NOW)["trade"]["trade_id"]
    blank = [""] * 12
    with open(cfg.PAPER_TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["settle", tid, NOW.isoformat()] + blank + ["yes", 1.0, 0.54, ""])
        w.writerow(["settle", tid, NOW.isoformat()] + blank + ["no", 0.0, -0.46, ""])      # a duplicate: ignored
        w.writerow(["settle", "ghost", NOW.isoformat()] + blank + ["yes", 1.0, 0.5, ""])   # no such trade: ignored
    (t,) = pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)
    assert (t["status"], t["outcome"], t["payout"], t["pnl"]) == ("settled", "yes", 1.0, 0.54)


# ---------- settling ----------

def test_a_resolved_contract_is_settled_with_the_real_result(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 10, now=NOW)        # 10 @ 44c, fee 0.18 -> pays 10 on yes
    stub_results(monkeypatch, {"KXBTC15M-T1": "yes"})
    assert pt.settle_due(cfg, now=CLOSE + timedelta(seconds=30)) == 1
    (t,) = pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)
    fee = pt.order_fee(0.44, 10)
    assert t["status"] == "settled" and t["outcome"] == "yes"
    assert t["pnl"] == round(10 - 4.4 - fee, 4)
    s = rows(cfg)[-1]
    assert s["event"] == "settle" and s["ticker"] == "" and s["side"] == ""        # only what a settle carries


def test_a_losing_position_loses_cost_and_fee(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    stub_results(monkeypatch, {"KXBTC15M-T1": "no"})
    pt.settle_due(cfg, now=CLOSE + timedelta(seconds=30))
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["pnl"] == -0.46      # 0.44 + 0.02 fee


def test_an_open_contract_is_not_asked_about(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    calls = []
    stub_results(monkeypatch, {"KXBTC15M-T1": "yes"}, calls)
    assert pt.settle_due(cfg, now=CLOSE - timedelta(seconds=1)) == 0
    assert calls == []


def test_closed_but_unresolved_stays_open_and_is_not_hammered(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    calls = []
    stub_results(monkeypatch, {"KXBTC15M-T1": None}, calls)                 # closed, no result yet
    t0 = CLOSE + timedelta(seconds=5)
    assert pt.settle_due(cfg, now=t0) == 0
    assert pt.settle_due(cfg, now=t0 + timedelta(seconds=5)) == 0           # inside the retry window
    assert calls == ["KXBTC15M-T1"]
    assert pt.settle_due(cfg, now=t0 + timedelta(seconds=25)) == 0          # window passed: asked again
    assert calls == ["KXBTC15M-T1"] * 2
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["status"] == "open"


def test_kalshi_being_unreachable_leaves_the_position_open(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    stub_results(monkeypatch, {})                                            # every call raises
    assert pt.settle_due(cfg, now=CLOSE + timedelta(minutes=1)) == 0
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["status"] == "open"


def test_settling_twice_never_writes_a_second_settle_row(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    stub_results(monkeypatch, {"KXBTC15M-T1": "yes"})
    later = CLOSE + timedelta(minutes=1)
    assert pt.settle_due(cfg, now=later) == 1
    assert pt.settle_due(cfg, now=later, retry_seconds=0) == 0
    # Two tabs racing: the second sees the trade still open in its own read, but the locked
    # duplicate check refuses the write.
    stale = pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)
    stale[0]["status"] = "open"
    monkeypatch.setattr(pt, "load_trades", lambda path: stale)
    assert pt.settle_due(cfg, now=later, retry_seconds=0) == 0
    assert sum(r["event"] == "settle" for r in rows(cfg)) == 1


def test_several_orders_in_one_market_cost_one_query_and_all_settle(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    pt.place_order(cfg, market(), "no", 2, now=NOW)
    calls = []
    stub_results(monkeypatch, {"KXBTC15M-T1": "no"}, calls)
    assert pt.settle_due(cfg, now=CLOSE + timedelta(seconds=30)) == 2
    assert calls == ["KXBTC15M-T1"]
    yes, no = pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)
    assert (yes["pnl"], no["payout"]) == (-0.46, 2.0)                        # the YES lost, the NO paid $2


# ---------- the summary ----------

def settled(ticker, side, contracts, price, fee, outcome):
    s = pt.settlement(side, contracts, price, fee, outcome)
    return {"status": "settled", "ticker": ticker, "side": side, "contracts": contracts, "price": price,
            "fee": fee, "outcome": outcome, "pnl": s["pnl"], "payout": s["payout"]}


THREE = [
    settled("A", "yes", 1, 0.40, 0.02, "yes"),      # +0.58
    settled("B", "no", 1, 0.60, 0.02, "yes"),       # -0.62
    settled("C", "yes", 2, 0.50, 0.04, "no"),       # -1.04
]


def test_summary_numbers():
    s = pt.summarize(THREE, min_sample=2, noise_z=2.0)
    assert (s["n_settled"], s["n_open"], s["n_markets"], s["wins"], s["losses"]) == (3, 0, 3, 1, 2)
    assert s["net_pnl"] == pytest.approx(-1.08)                 # 0.58 - 0.62 - 1.04
    assert s["fees"] == pytest.approx(0.08)
    assert s["staked"] == pytest.approx(2.08)                   # (0.40+0.02) + (0.60+0.02) + (1.00+0.04)
    assert s["win_rate"] == pytest.approx(0.25)                 # 1 of 4 contracts won
    assert s["breakeven_rate"] == pytest.approx(0.52)           # 2.08 / 4 contracts
    assert s["win_rate_se"] == pytest.approx(0.25)              # sqrt(0.25 * 0.75 / 3 markets)
    assert s["roi"] == pytest.approx(-1.08 / 2.08)


def test_summary_verdicts():
    assert pt.summarize(THREE)["verdict"] == "small_sample"                         # 3 < 30 markets
    assert pt.summarize(THREE, min_sample=2, noise_z=2.0)["verdict"] == "within_noise"   # gap 0.27 < 2 * 0.25
    assert pt.summarize(THREE, min_sample=2, noise_z=0.5)["verdict"] == "behind"          # gap 0.27 > 0.5 * 0.25
    assert pt.summarize([])["verdict"] == "none"


def test_many_winners_in_a_large_sample_is_ahead():
    wins = [settled(f"M{i}", "yes", 1, 0.50, 0.02, "yes" if i % 10 else "no") for i in range(100)]   # 90% wins
    s = pt.summarize(wins, min_sample=30, noise_z=2.0)
    assert s["verdict"] == "ahead" and s["win_rate"] == pytest.approx(0.9)


def test_orders_in_one_market_count_as_one_independent_market():
    s = pt.summarize([settled("A", "yes", 1, 0.5, 0.02, "yes"), settled("A", "yes", 1, 0.5, 0.02, "yes"),
                      settled("B", "yes", 1, 0.5, 0.02, "no")], min_sample=1)
    assert s["n_settled"] == 3 and s["n_markets"] == 2


def test_open_positions_are_counted_but_do_not_move_the_account():
    s = pt.summarize([{"status": "open"}, *THREE])
    assert s["n_open"] == 1 and s["net_pnl"] == pytest.approx(-1.08)


def test_report_states_the_limits_every_time(tmp_path):
    cfg = _cfg(tmp_path)
    assert pt.LIMITS_NOTE in pt.report(cfg)                       # even with an empty ledger
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    text = pt.report(cfg)
    assert "1 placed, 1 open, 0 settled" in text and pt.LIMITS_NOTE in text


# ---------- who placed it ----------

def test_every_row_is_as_wide_as_the_header_and_manual_is_the_default_source(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    stub_results(monkeypatch, {"KXBTC15M-T1": "yes"})
    pt.settle_due(cfg, now=CLOSE + timedelta(seconds=30))
    with open(cfg.PAPER_TRADE_LOG_PATH, newline="") as f:
        assert {len(r) for r in csv.reader(f)} == {len(pt.LEDGER_HEADER)}
    assert rows(cfg)[0]["source"] == "manual"
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["source"] == "manual"


def test_a_row_with_a_blank_source_reads_as_manual(tmp_path):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    text = (tmp_path / "paper.csv").read_text().replace(",manual", ",")
    (tmp_path / "paper.csv").write_text(text)
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["source"] == "manual"


# ---------- the automatic rule ----------

@pytest.mark.parametrize("composite, strength, threshold, side", [
    ("bullish", 3, 3, "yes"),
    ("bearish", 5, 3, "no"),
    ("bullish", 2, 3, None),            # leaning, but not strong enough to alert
    ("bearish", 2, 3, None),
    ("neutral", 0, 3, None),
    ("neutral", 9, 3, None),            # neutral never trades, whatever the number says
])
def test_composite_rule(composite, strength, threshold, side):
    assert pt.decide("composite", composite, strength, threshold)[0] == side


@pytest.mark.parametrize("p, side", [
    (0.70, "yes"),      # 0.70 - 0.44 - 0.02 fee = 0.24 > margin
    (0.30, "no"),       # 0.70 - 0.57 - 0.02 fee = 0.11 > margin
    (0.50, "yes"),      # 0.50 - 0.44 - 0.02 fee = 0.04 > 0.02 margin
    (0.46, None),       # 0.46 - 0.44 - 0.02 = 0: the spread and the fee eat a 2-point lean
    (0.45, None),
])
def test_model_rule_by_hand(p, side):
    assert pt.decide("model", model_p_yes=p, yes_bid=0.43, yes_ask=0.44, margin=0.02)[0] == side


def test_model_rule_is_the_edge_studys_rule_over_random_quotes():
    rng = np.random.default_rng(7)
    for _ in range(500):
        bid = round(float(rng.uniform(0.02, 0.9)), 2)
        ask = round(min(0.98, bid + float(rng.uniform(0.01, 0.1))), 2)
        p, margin, n = float(rng.uniform(0, 1)), float(rng.choice([0.0, 0.02, 0.05])), int(rng.choice([1, 5, 100]))
        study, _ = kalshi_study.model_trades(p, bid, ask, 0, margin, n)
        want = {1: "yes", -1: "no", 0: None}[int(study)]
        assert pt.decide("model", model_p_yes=p, yes_bid=bid, yes_ask=ask, contracts=n, margin=margin)[0] == want


def test_model_rule_without_a_probability_or_quote_does_not_trade():
    assert pt.decide("model", yes_bid=0.4, yes_ask=0.5)[0] is None
    assert pt.decide("model", model_p_yes=0.9, yes_bid=None, yes_ask=0.5)[0] is None


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        pt.decide("coinflip")


# ---------- auto_step ----------

def step(cfg, minutes_left, live=None, fetch=None, strategy="composite", **kw):
    now = CLOSE - timedelta(minutes=minutes_left)
    live = live or market()
    kw = {"composite": "bullish", "strength": 4, "alert_threshold": 3, **kw}
    return pt.auto_step(cfg, live, fetch or (lambda: live), strategy, 1, now=now, **kw)


@pytest.mark.parametrize("minutes_left, action", [
    (7.0, "waiting"), (5.01, "waiting"),           # before the window
    (5.0, "traded"), (3.0, "traded"), (2.0, "traded"),
    (1.99, "waiting"), (0.5, "waiting"),           # after it: too late to be the rule's entry
])
def test_entry_window(tmp_path, minutes_left, action):
    cfg = _cfg(tmp_path)
    assert step(cfg, minutes_left)["action"] == action
    if action == "traded":
        assert len(rows(cfg)) == 1
    else:
        assert not (tmp_path / "paper.csv").exists()


def test_a_traded_window_is_tagged_with_its_rule_and_the_context(tmp_path):
    cfg = _cfg(tmp_path)
    res = step(cfg, 4.0, strategy="composite", spot=75_100.0, model_p_yes=0.4)
    assert res["action"] == "traded" and res["trade"]["source"] == "auto:composite"
    (o,) = rows(cfg)
    assert (o["source"], o["side"], o["composite"], o["spot"], o["price"]) == ("auto:composite", "yes", "bullish", "75100.0", "0.44")
    assert pt.load_trades(cfg.PAPER_TRADE_LOG_PATH)[0]["source"] == "auto:composite"


def test_it_trades_a_market_once_however_often_it_is_called(tmp_path):
    cfg = _cfg(tmp_path)
    assert step(cfg, 4.5)["action"] == "traded"
    for left in (4.0, 3.0, 2.5):
        assert step(cfg, left)["action"] == "done"
    assert len(rows(cfg)) == 1


def test_a_different_rule_chosen_mid_window_does_not_double_up(tmp_path):
    cfg = _cfg(tmp_path)
    assert step(cfg, 4.0, strategy="composite")["action"] == "traded"
    assert step(cfg, 3.5, strategy="model", model_p_yes=0.9)["action"] == "done"
    assert len(rows(cfg)) == 1


def test_a_new_market_gets_its_own_trade(tmp_path):
    cfg = _cfg(tmp_path)
    assert step(cfg, 4.0)["action"] == "traded"
    nxt = market(ticker="KXBTC15M-T2", close_time=CLOSE + timedelta(minutes=15))
    assert pt.auto_step(cfg, nxt, lambda: nxt, "composite", 1, composite="bearish", strength=4,
                        now=CLOSE + timedelta(minutes=10))["action"] == "traded"
    assert [r["side"] for r in rows(cfg)] == ["yes", "no"]


def test_no_signal_means_no_order_and_no_extra_quote_fetch(tmp_path):
    cfg = _cfg(tmp_path)
    fetched = []
    res = step(cfg, 4.0, fetch=lambda: fetched.append(1) or market(), composite="neutral", strength=0)
    assert res["action"] == "no_signal" and fetched == []
    assert not (tmp_path / "paper.csv").exists()
    # and it keeps looking: the signal appearing later in the window still trades
    assert step(cfg, 3.0)["action"] == "traded"


def test_the_order_is_priced_on_the_fresh_quote(tmp_path):
    cfg = _cfg(tmp_path)
    moved = market(yes_bid=0.50, yes_ask=0.52)
    step(cfg, 4.0, fetch=lambda: moved)
    assert rows(cfg)[0]["price"] == "0.52"


def test_a_stale_quote_cannot_place_a_trade_the_fresh_one_would_not(tmp_path):
    cfg = _cfg(tmp_path)
    live = market(yes_bid=0.43, yes_ask=0.44)             # model 70%: a clear YES on this quote
    moved = market(yes_bid=0.68, yes_ask=0.69)            # the market caught up: no edge left
    res = step(cfg, 4.0, live=live, fetch=lambda: moved, strategy="model", model_p_yes=0.70)
    assert res["action"] == "no_signal" and "fresh quote" in res["reason"]
    assert not (tmp_path / "paper.csv").exists()


def test_two_tabs_deciding_at_once_place_one_order(tmp_path):
    cfg = _cfg(tmp_path)
    # Both tabs passed the has_auto_trade() pre-check before either wrote; the locked check decides.
    first = pt.place_order(cfg, market(), "yes", 1, now=NOW, source="auto:composite", once_per_market=True)
    second = pt.place_order(cfg, market(), "yes", 1, now=NOW, source="auto:composite", once_per_market=True)
    assert first["ok"] and not second["ok"] and second["duplicate"]
    assert len(rows(cfg)) == 1


def test_manual_orders_are_never_blocked_by_an_automatic_one(tmp_path):
    cfg = _cfg(tmp_path)
    step(cfg, 4.0)
    assert pt.place_order(cfg, market(), "no", 1, now=CLOSE - timedelta(minutes=3))["ok"]
    assert [r["source"] for r in rows(cfg)] == ["auto:composite", "manual"]


def test_no_market_is_reported_not_traded(tmp_path):
    cfg = _cfg(tmp_path)
    res = pt.auto_step(cfg, {"available": False, "reason": "No open KXBTC15M market found right now."},
                       lambda: None, "composite", 1, composite="bullish", strength=5, now=NOW)
    assert res["action"] == "blocked" and "No open" in res["reason"]


# ---------- results by who placed the trade ----------

def test_the_account_is_read_separately_for_manual_and_auto_trades():
    trades = [dict(settled("A", "yes", 1, 0.40, 0.02, "yes"), source="manual"),       # +0.58
              dict(settled("B", "yes", 1, 0.40, 0.02, "no"), source="auto:composite"),   # -0.42
              dict(settled("C", "no", 1, 0.60, 0.02, "no"), source="auto:composite")]    # +0.38
    by = pt.summarize_by_source(trades, min_sample=1)
    assert list(by) == ["manual", "auto:composite"]
    assert by["manual"]["net_pnl"] == pytest.approx(0.58) and by["manual"]["n_settled"] == 1
    assert by["auto:composite"]["net_pnl"] == pytest.approx(-0.04) and by["auto:composite"]["n_markets"] == 2
    assert by["auto:composite"]["win_rate"] == pytest.approx(0.5)
    assert pt.summarize(trades)["net_pnl"] == pytest.approx(0.54)                        # the whole account is unchanged


def test_report_lists_each_source(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pt.place_order(cfg, market(), "yes", 1, now=NOW)
    pt.place_order(cfg, market(ticker="KXBTC15M-T2"), "yes", 1, now=NOW, source="auto:model")
    text = pt.report(cfg)
    assert "manual: 1 open" in text and "auto:model: 1 open" in text
