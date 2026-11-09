"""
Live probe of Kalshi's public market-data endpoints (DASHBOARD_NOTES.md).

1. One GET of the open-markets list and one of /markets/{ticker}/orderbook: saves
   their full response headers and body sizes to tests/fixtures/kalshi_response_headers.json,
   and checks the two agree on top of book (back to back, so they can differ by a tick).
2. Polls the orderbook once a second for `seconds` (default 60), sequentially, one
   connection: records status codes, 429s, Retry-After and any rate-limit headers.
No auth, plain requests, no header tricks. Prints <= 30 lines.
"""
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kalshi_api  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures",
                       "kalshi_response_headers.json")


def main(seconds: int = 60, interval: float = 1.0):
    s = requests.Session()
    t0 = time.time()
    rm = s.get(f"{kalshi_api.BASE_URL}/markets", params={"series_ticker": kalshi_api.SERIES_TICKER, "status": "open"},
               timeout=10)
    rm.raise_for_status()
    market = rm.json()["markets"][0]
    ticker = market["ticker"]
    ro = s.get(f"{kalshi_api.BASE_URL}/markets/{ticker}/orderbook", params={"depth": 1}, timeout=10)
    ro.raise_for_status()
    rs = s.get(f"{kalshi_api.BASE_URL}/markets/{ticker}", timeout=10)
    rs.raise_for_status()
    top = kalshi_api.parse_orderbook(ro.json())
    rec = kalshi_api.parse_market(rs.json()["market"])
    cap = {"captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "ticker": ticker,
           "markets_list": {"status": rm.status_code, "bytes": len(rm.content), "headers": dict(rm.headers)},
           "orderbook_depth1": {"status": ro.status_code, "bytes": len(ro.content), "headers": dict(ro.headers),
                                "body": ro.json()},
           "market_record": {"status": rs.status_code, "bytes": len(rs.content), "headers": dict(rs.headers)}}
    with open(FIXTURE, "w", encoding="utf-8") as f:
        json.dump(cap, f, indent=1, sort_keys=True)
    print(f"ticker {ticker}; bytes: list {len(rm.content)}, market {len(rs.content)}, orderbook(depth=1) {len(ro.content)}")
    print(f"top of book: orderbook bid/ask {top['yes_bid']}/{top['yes_ask']}  market record {rec['yes_bid']}/{rec['yes_ask']}")
    rl = sorted(k for h in (rm.headers, ro.headers, rs.headers) for k in h if "rate" in k.lower() or "retry" in k.lower()
                or "limit" in k.lower())
    print(f"rate-limit / retry headers seen: {rl or 'none'}")
    print(f"orderbook headers: {sorted(ro.headers)}")

    codes, lat, hits, extra = {}, [], [], set()
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        start = time.time()
        try:
            r = s.get(f"{kalshi_api.BASE_URL}/markets/{ticker}/orderbook", params={"depth": 1}, timeout=10)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if r.status_code == 429:
                hits.append((round(start - t0, 1), r.headers.get("Retry-After")))
            extra |= {k for k in r.headers if "rate" in k.lower() or "retry" in k.lower()}
        except requests.RequestException as e:
            codes[type(e).__name__] = codes.get(type(e).__name__, 0) + 1
        lat.append(time.time() - start)
        n += 1
        time.sleep(max(0.0, interval - (time.time() - start)))
    print(f"poll: {n} requests in {seconds}s at {interval}s; status {codes}; 429s {hits or 'none'}")
    print(f"latency median {statistics.median(lat) * 1000:.0f} ms, p90 {sorted(lat)[int(0.9 * len(lat))] * 1000:.0f} ms, "
          f"max {max(lat) * 1000:.0f} ms; rate headers during poll: {sorted(extra) or 'none'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
