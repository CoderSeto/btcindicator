"""Live timing of the dashboard's strip work (price + Kalshi quote + decision), at its own interval.
One page refresh's state is loaded once; then the strip's compute path runs for `seconds`. Prints <= 10 lines."""
import os, statistics, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, dashboard_ui as ui, data_fetch, kalshi, kalshi_api, kalshi_decision  # noqa: E401,E402
from dashboard_data import load_state  # noqa: E402

cfg = SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
cfg.ENABLE_EXCHANGE_FEEDS = cfg.ENABLE_BOOK_FEEDS = False
seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
state = load_state(cfg)
ticker_calls = []
_orig = data_fetch.get_ticker
data_fetch.get_ticker = lambda p: ticker_calls.append(1) or _orig(p)
m = kalshi_api.live_market(cfg)
rule, res = kalshi_decision.load_active(cfg.DECISION_BACKTEST_PATH)
ctx = {"product_id": cfg.PRODUCT_ID, "candles": state.df, "price_df": state.price_df, "auto_track": True,
       "strike": m["strike"], "minutes": m["minutes_remaining"], "rule": rule, "rule_result": res, "open_trades": []}
p = kalshi_api.shared_poller(0, 0, 0)
r0, times, ages, actions = p.requests, [], [], {}
end = time.time() + seconds
while time.time() < end:
    t = time.perf_counter()
    price = ui.shared_latest_price(cfg.PRODUCT_ID, cfg.STRIP_REFRESH_SECONDS / 2, state.df)
    v = ui.live_view(cfg, ctx, kalshi_api.live_market(cfg), price)
    times.append(time.perf_counter() - t)
    ages.append(v["quote_age"] or 0.0)
    a = v["decision"]["action"] if v["decision"] else "none"
    actions[a] = actions.get(a, 0) + 1
    time.sleep(max(0.0, cfg.STRIP_REFRESH_SECONDS - (time.perf_counter() - t)))
mins = seconds / 60
print(f"strip refreshes {len(times)} in {seconds}s; work per refresh median {statistics.median(times) * 1000:.0f} ms, "
      f"p90 {sorted(times)[int(0.9 * len(times))] * 1000:.0f} ms")
print(f"requests/min: Coinbase ticker {len(ticker_calls) / mins:.0f}, Kalshi {(p.requests - r0) / mins:.0f} "
      f"(poll interval now {p.interval:.2f}s, last 429 {p.last_429})")
print(f"quote age at render: median {statistics.median(ages):.2f}s, max {max(ages):.2f}s; actions {actions}")
