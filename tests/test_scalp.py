"""scalp.py: values worked by hand from Kalshi's fee table (ceil(7% * p * (1 - p)) per one-contract order)."""
import pytest

import kalshi
import scalp

TRADE = {"side": "yes", "contracts": 1, "price": 0.53, "fee": 0.02}        # 55c in


def test_breakeven_sale_by_hand():
    # sale 57c: fee ceil(7% * .57 * .43 = 1.7c) = 2c -> .57 - .02 = .55 >= .55 (breaks even); 56c nets -1c
    assert scalp.breakeven_sale_cents(TRADE) == 57


def test_position_view_by_hand():
    v = scalp.position_view(TRADE, 0.78, 0.79)
    assert v["sale_price"] == pytest.approx(0.78) and v["sale_fee"] == pytest.approx(0.02)
    assert v["net_pnl"] == pytest.approx(0.21) and v["breakeven_c"] == 57 and v["needed_c"] == 0.0
    v = scalp.position_view(TRADE, 0.52, 0.53)                         # unchanged quote: 5c short of... needs 5c
    assert v["net_pnl"] == pytest.approx(-0.05) and v["needed_c"] == pytest.approx(5.0)


def test_no_position_sells_at_one_minus_the_ask():
    no = {"side": "no", "contracts": 1, "price": 0.48, "fee": 0.02}
    v = scalp.position_view(no, 0.40, 0.42)
    assert v["sale_price"] == pytest.approx(0.58) and v["net_pnl"] == pytest.approx(0.06)


def test_no_exit_quote_gives_none_not_zero():
    v = scalp.position_view(TRADE, None, 0.53)
    assert v["sale_price"] is None and v["net_pnl"] is None and v["needed_c"] is None


def test_exit_signal_thresholds_are_net_cents_per_contract():
    hi = scalp.position_view(TRADE, 0.78, 0.79)                        # +21c
    lo = scalp.position_view(TRADE, 0.40, 0.41)                        # -17c
    mid = scalp.position_view(TRADE, 0.55, 0.56)                       # 55-2-55 = -2c
    assert scalp.exit_signal(TRADE, hi, 6, 6) == "TARGET"
    assert scalp.exit_signal(TRADE, lo, 6, 6) == "STOP"
    assert scalp.exit_signal(TRADE, mid, 6, 6) is None
    assert scalp.exit_signal(TRADE, {"net_pnl": None}, 6, 6) is None


def test_multi_contract_breakeven_uses_the_order_fee():
    t = {"side": "yes", "contracts": 100, "price": 0.50, "fee": 1.75}   # 100 at 50c: fee $1.75
    # entry cost 51.75; sale 52c on 100: fee ceil(7*100*.52*.48 = 1.7472) -> $1.75; 52 - 1.75 = 50.25 < 51.75
    c = scalp.breakeven_sale_cents(t)
    assert c is not None and c > 52


@pytest.mark.parametrize("side", ["yes", "no"])
def test_needed_btc_move_moves_the_model_by_the_asked_cents(side):
    spot, strike, minutes, vol = 75_050.0, 75_041.0, 8.0, 0.70
    move = scalp.needed_btc_move(spot, strike, minutes, vol, side, 5.0)
    assert move is not None and move > 0
    p0 = kalshi.settlement_probability(spot, strike, minutes, vol)
    p1 = kalshi.settlement_probability(spot + (move if side == "yes" else -move), strike, minutes, vol)
    assert abs(p1 - p0) == pytest.approx(0.05, abs=1e-4) and (p1 > p0) == (side == "yes")


def test_needed_move_grows_with_the_cents_and_is_none_when_unreachable():
    a = scalp.needed_btc_move(75_050.0, 75_041.0, 8.0, 0.70, "yes", 5.0)
    b = scalp.needed_btc_move(75_050.0, 75_041.0, 8.0, 0.70, "yes", 10.0)
    assert b > a
    assert scalp.needed_btc_move(75_050.0, 75_041.0, 8.0, 0.70, "yes", 60.0) is None   # past 100%
    assert scalp.needed_btc_move(None, 75_041.0, 8.0, 0.70, "yes", 5.0) is None
    assert scalp.needed_btc_move(75_050.0, 75_041.0, 0.0, 0.70, "yes", 5.0) is None
