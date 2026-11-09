# Dashboard live strip and Kalshi polling (measured 2026-11-08, user's machine)

| probe (`review/kalshi_poll_probe.py`, `review/strip_timing.py`) | result |
|---|---|
| orderbook `?depth=1` at 1.0 s for 60 s | 60/60 HTTP 200, no 429, median 16 ms, max 24 ms |
| orderbook `?depth=1` at 0.5 s for 60 s | 120/120 HTTP 200, no 429 |
| rate-limit / Retry-After headers | none on any endpoint (full headers: `tests/fixtures/kalshi_response_headers.json`) |
| body size | orderbook depth=1: 91 B; market record: 2,135 B; open-markets list: 2,151 B |
| market record vs orderbook top of book | 0 of 15 back-to-back reads agreed; the list held one quote ~20 s while the book moved 8c |
| highest rate tested and sustained | 2 requests/s (0.5 s). Higher was not tried; no limit is assumed |
| strip work per refresh (price + quote + decision) | median 158 ms, p90 179 ms |
| strip request rate per process | Coinbase ticker 60/min, Kalshi 60/min at a 1 s strip, shared by all tabs |
| quote age when rendered | median 0.00 s, max 0.08 s |

Defaults set from this: `KALSHI_POLL_MIN_SECONDS = 0.5` (the measured sustained rate), `KALSHI_POLL_MAX_SECONDS = 10`,
`STRIP_REFRESH_SECONDS = 1`, `KALSHI_MAX_QUOTE_AGE_SECONDS = 5`, `KALSHI_MARKET_REFRESH_SECONDS = 30`.
Quote comes from the orderbook; the market record is read only for ticker, strike and close time.
Backoff: 429 / 5xx / timeout doubles the interval or waits Retry-After, and nothing is sent until it passes.
Screenshots: skipped, Playwright is not installed on this machine.
