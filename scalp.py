"""
Scalping helpers for KXBTC15M paper positions: what a position is worth if sold back
now (real bid, fee again), the sale price that breaks even, and how far BTC has to move
for that. Pure functions over a paper_trading position dict; no network.

A scalp pays the spread and the taker fee on the way in AND on the way out (the scalp
study measured ~4-5c per round trip), so a position starts underwater by that much.
The needed-BTC-move figure uses the settlement model (kalshi.settlement_probability) as
a ruler for how many dollars of BTC move the contract's price by N cents; it says how big
a move is needed, never which way BTC will go.
"""
import kalshi
import paper_trading

EPS = 1e-9


def entry_cost(trade: dict) -> float:
    """Everything paid to open: the contracts plus the entry fee."""
    return trade["price"] * trade["contracts"] + trade["fee"]


def breakeven_sale_cents(trade: dict):
    """Lowest whole-cent sale price per contract that nets >= 0 after the exit fee, or None if none does."""
    n = trade["contracts"]
    for cents in range(1, 100):
        p = cents / 100.0
        if p * n - paper_trading.order_fee(p, n) - entry_cost(trade) >= -EPS:
            return cents
    return None


def position_view(trade: dict, yes_bid, yes_ask) -> dict:
    """
    {"sale_price", "sale_fee", "net_pnl", "breakeven_c", "needed_c"} for selling `trade` back now.
    `needed_c` is how many cents the sale price must still rise to break even (0 if it already
    has). Everything is None when the exit side has no quote.
    """
    sale = paper_trading.sale_price(trade["side"], yes_bid, yes_ask)
    be = breakeven_sale_cents(trade)
    out = {"sale_price": sale, "sale_fee": None, "net_pnl": None, "breakeven_c": be, "needed_c": None}
    if sale is None:
        return out
    fee = paper_trading.order_fee(sale, trade["contracts"])
    out["sale_fee"] = fee
    out["net_pnl"] = round(sale * trade["contracts"] - fee - entry_cost(trade), 4)
    if be is not None:
        out["needed_c"] = max(0.0, be - sale * 100.0)
    return out


def exit_signal(trade: dict, view: dict, target_cents: float, stop_cents: float):
    """"TARGET" when net P&L per contract has reached +target_cents, "STOP" at -stop_cents, else None."""
    if view["net_pnl"] is None:
        return None
    per = view["net_pnl"] / trade["contracts"] * 100.0
    if per >= target_cents - EPS:
        return "TARGET"
    if per <= -stop_cents + EPS:
        return "STOP"
    return None


def needed_btc_move(spot, strike, minutes, annual_vol, side: str, cents: float, iterations: int = 60):
    """
    Dollars of BTC, in the direction this position wants (up for YES, down for NO), that move the
    MODEL's P(YES) by `cents` toward the position. None if the model can't reach that far or an
    input is missing. A ruler for the size of the move, not a forecast.
    """
    if None in (spot, strike, minutes, annual_vol) or cents <= 0 or minutes <= 0:
        return None
    try:
        p0 = kalshi.settlement_probability(spot, strike, minutes, annual_vol)
        sign = 1.0 if side == "yes" else -1.0
        goal = p0 + sign * cents / 100.0
        if not 0.0 < goal < 1.0:
            return None
        lo, hi = 0.0, spot * 0.05
        f = lambda m: kalshi.settlement_probability(spot + sign * m, strike, minutes, annual_vol)  # noqa: E731
        reached = f(hi) >= goal if sign > 0 else f(hi) <= goal
        if not reached:
            return None
        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            ok = f(mid) >= goal if sign > 0 else f(mid) <= goal
            lo, hi = (lo, mid) if ok else (mid, hi)
        return hi
    except ValueError:
        return None
