"""
The console loop (main.run) and the Kalshi CLI (python kalshi.py) read the live
price the way the dashboard does: from Coinbase's ticker, because the candles
endpoint is cached ~5 minutes (measured from the user's machine, 2026-11-07),
and with a printed warning when they have to fall back to the newest candle.

Both entry points are driven for real: main.run() for one pass of its loop
(the sleep at the end of the pass raises KeyboardInterrupt, which is how the
loop stops), and kalshi.py through runpy as __main__. Only Coinbase is stubbed.
"""
import runpy
import sys
from types import SimpleNamespace

import pytest

import config as cfg
import data_fetch
import main
from test_dashboard_data import REST_BOOK, TICKER, candles


@pytest.fixture
def coinbase(monkeypatch, tmp_path):
    state = SimpleNamespace(ticker=TICKER, df=candles(200))
    monkeypatch.setattr(data_fetch, "get_candles", lambda *a, **k: state.df.copy())
    monkeypatch.setattr(data_fetch, "get_order_book", lambda *a, **k: REST_BOOK)

    def get_ticker(product_id):
        if state.ticker is None:
            raise RuntimeError("ticker down")
        return dict(state.ticker)
    monkeypatch.setattr(data_fetch, "get_ticker", get_ticker)
    for name, value in (("ENABLE_EXCHANGE_FEEDS", False), ("ENABLE_BOOK_FEEDS", False),
                        ("LOG_ALL_SIGNALS", False), ("ALERT_LOG_PATH", str(tmp_path / "a.csv")),
                        ("SIGNAL_LOG_PATH", str(tmp_path / "s.csv")),
                        ("BACKTEST_RESULTS_PATH", str(tmp_path / "bt.json"))):
        monkeypatch.setattr(cfg, name, value)
    return state


def _one_console_pass(monkeypatch, capsys):
    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(main, "time", SimpleNamespace(sleep=stop, time=__import__("time").time))
    main.run()
    return capsys.readouterr().out


def test_console_status_line_shows_the_ticker_price(coinbase, monkeypatch, capsys):
    out = _one_console_pass(monkeypatch, capsys)
    assert "$75,712.34" in out
    assert f"${coinbase.df['close'].iloc[-1]:,.2f}" not in out
    assert "newest candle" not in out


def test_console_falls_back_to_the_candle_and_says_it_may_be_stale(coinbase, monkeypatch, capsys):
    coinbase.ticker = None
    out = _one_console_pass(monkeypatch, capsys)
    assert f"${coinbase.df['close'].iloc[-1]:,.2f}" in out
    assert "ticker down" in out and "5 minutes" in out
    assert "[error]" not in out


def _run_kalshi_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["kalshi.py", *args])
    runpy.run_path(str(__import__("pathlib").Path(main.__file__).with_name("kalshi.py")), run_name="__main__")
    return capsys.readouterr().out


def test_kalshi_cli_prices_the_ticker_spot(coinbase, monkeypatch, capsys):
    out = _run_kalshi_cli(monkeypatch, capsys, "--strike", "75700", "--minutes", "5")
    assert "Current price:       $75,712.34  (from ticker)" in out
    assert "!!" not in out


def test_kalshi_cli_warns_when_it_falls_back_to_the_candle(coinbase, monkeypatch, capsys):
    coinbase.ticker = None
    out = _run_kalshi_cli(monkeypatch, capsys, "--strike", "75700", "--minutes", "5")
    assert f"${coinbase.df['close'].iloc[-1]:,.2f}  (from candles)" in out
    assert "!! ticker unavailable (ticker down)" in out and "5 minutes" in out
