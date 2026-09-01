import time
import sys
sys.path.insert(0, ".")
import exchange_feeds as ef

now_ms = int(time.time() * 1000)
start_ms = now_ms - 24 * 3600 * 1000
print(f"now_ms={now_ms}  start_ms={start_ms}")
print(f"now readable: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now_ms/1000))}")
print(f"start readable: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(start_ms/1000))}")

trades = ef.fetch_binance_us_trade_history("BTCUSDT", start_ms, now_ms, pause=0.3, max_requests=5)
print(f"Got {len(trades)} trades")
if trades:
    print("first:", trades[0])
    print("last:", trades[-1])
