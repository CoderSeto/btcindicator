"""
Tests for indicators.py.

Every indicator is checked against an independent, deliberately plain
pure-Python reference (loops and the statistics module, no pandas rolling /
ewm), plus small hand-worked cases whose answers can be verified on paper.

Behaviours worth knowing, pinned here so a change is a decision, not an accident:
  - RSI uses a *simple* rolling mean of gains and losses (Cutler's RSI), not
    Wilder smoothing, so values differ from most charting platforms.
  - RSI is 50 whenever it is undefined -- including a window with gains and
    no losses at all (true RSI would be 100) and the first `period` candles.
  - The volume average includes the current candle, so a spike has to clear
    a threshold that it itself helps raise.
"""
import random
import statistics

import numpy as np
import pandas as pd
import pytest

import config as real_cfg
import indicators as ind


def frame(closes, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "close": [float(c) for c in closes],
        "volume": [float(v) for v in (volumes if volumes is not None else [1.0] * n)],
    })


def random_closes(n=200, seed=5):
    rng = np.random.default_rng(seed)
    return list(100 + np.cumsum(rng.normal(0, 1, n)))


# ---------- pure-python references ----------

def ref_sma(xs, p):
    return [None if i + 1 < p else statistics.fmean(xs[i + 1 - p:i + 1]) for i in range(len(xs))]


def ref_ema(xs, span):
    a = 2.0 / (span + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out


def ref_rsi(xs, p):
    out = []
    for t in range(len(xs)):
        if t < p:
            out.append(50.0)       # need p diffs, which first exist at t == p
            continue
        diffs = [xs[k] - xs[k - 1] for k in range(t - p + 1, t + 1)]
        ag = sum(d for d in diffs if d > 0) / p
        al = sum(-d for d in diffs if d < 0) / p
        out.append(50.0 if al == 0 else 100 - 100 / (1 + ag / al))
    return out


def assert_matches(series, ref, tol=1e-9):
    assert len(series) == len(ref)
    for i, (got, want) in enumerate(zip(series, ref)):
        if want is None:
            assert pd.isna(got), f"index {i}: expected NaN warmup, got {got}"
        else:
            assert got == pytest.approx(want, abs=tol, rel=tol), f"index {i}"


# ---------- SMA ----------

@pytest.mark.parametrize("period", [1, 3, 9, 21])
def test_sma_matches_reference(period):
    xs = random_closes()
    assert_matches(ind.add_sma(frame(xs), period)[f"sma_{period}"], ref_sma(xs, period))


def test_sma_hand_computed_and_warmup():
    out = ind.add_sma(frame([2, 4, 6, 8, 10]), 3)["sma_3"]
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.dropna().tolist() == [4.0, 6.0, 8.0]


def test_sma_other_column():
    df = pd.DataFrame({"close": [1.0, 1, 1], "volume": [3.0, 6.0, 9.0]})
    assert ind.add_sma(df, 2, col="volume")["sma_2"].tolist()[1:] == [4.5, 7.5]


# ---------- EMA ----------

@pytest.mark.parametrize("span", [3, 12, 26])
def test_ema_matches_recursion(span):
    xs = random_closes()
    assert_matches(ind.add_ema(frame(xs), span)[f"ema_{span}"], ref_ema(xs, span))


def test_ema_hand_computed():
    # span 3 -> alpha 0.5: 10, 0.5*20+0.5*10, 0.5*30+0.5*15
    assert ind.add_ema(frame([10, 20, 30]), 3)["ema_3"].tolist() == [10.0, 15.0, 22.5]


def test_ema_of_constant_is_constant_and_starts_with_no_warmup_gap():
    out = ind.add_ema(frame([7] * 50), 12)["ema_12"]
    assert not out.isna().any() and (out == 7.0).all()


# ---------- RSI ----------

@pytest.mark.parametrize("period", [2, 5, 14])
def test_rsi_matches_reference(period):
    xs = random_closes()
    assert_matches(ind.add_rsi(frame(xs), period)["rsi"], ref_rsi(xs, period))


def test_rsi_hand_computed():
    out = ind.add_rsi(frame([10, 11, 10, 12]), 2)["rsi"].tolist()
    # t=2: gains [1,0] losses [0,1] -> 0.5/0.5 -> RS 1 -> 50
    # t=3: gains [0,2] losses [1,0] -> 1.0/0.5 -> RS 2 -> 100 - 100/3
    assert out[:2] == [50.0, 50.0]
    assert out[2] == pytest.approx(50.0)
    assert out[3] == pytest.approx(100 - 100 / 3)


def test_rsi_all_losses_is_zero():
    out = ind.add_rsi(frame([10, 9, 8, 7, 6, 5]), 3)["rsi"]
    assert out.iloc[-1] == 0.0


def test_rsi_all_gains_is_reported_as_neutral_50_not_100():
    out = ind.add_rsi(frame([1, 2, 3, 4, 5, 6]), 3)["rsi"]
    assert (out == 50.0).all()            # documented fallback: undefined RS -> neutral


def test_rsi_flat_series_is_50():
    assert (ind.add_rsi(frame([5] * 20), 14)["rsi"] == 50.0).all()


def test_rsi_bounded_and_never_nan():
    out = ind.add_rsi(frame(random_closes(1000, seed=9)), 14)["rsi"]
    assert not out.isna().any()
    assert out.between(0, 100).all()


def test_rsi_first_period_candles_are_neutral():
    xs = [100, 90, 80, 70, 60, 50, 40, 30]                # steadily falling
    out = ind.add_rsi(frame(xs), 4)["rsi"].tolist()
    assert out[:4] == [50.0] * 4 and out[4] == 0.0


# ---------- MACD ----------

def test_macd_matches_reference():
    xs = random_closes()
    out = ind.add_macd(frame(xs), 12, 26, 9)
    macd = [f - s for f, s in zip(ref_ema(xs, 12), ref_ema(xs, 26))]
    sig = ref_ema(macd, 9)
    assert_matches(out["macd"], macd)
    assert_matches(out["macd_signal"], sig)
    assert_matches(out["macd_hist"], [m - s for m, s in zip(macd, sig)])


def test_macd_starts_at_zero_and_is_zero_for_constant_price():
    out = ind.add_macd(frame(random_closes()), 12, 26, 9)
    assert out["macd"].iloc[0] == 0.0 and out["macd_signal"].iloc[0] == 0.0
    flat = ind.add_macd(frame([50] * 40), 12, 26, 9)
    assert (flat[["macd", "macd_signal", "macd_hist"]] == 0.0).all().all()


def test_macd_sign_follows_trend():
    up = ind.add_macd(frame(range(1, 120)), 12, 26, 9)
    assert up["macd"].iloc[-1] > 0
    down = ind.add_macd(frame(range(120, 1, -1)), 12, 26, 9)
    assert down["macd"].iloc[-1] < 0


# ---------- Bollinger ----------

def test_bollinger_matches_reference_using_sample_std():
    xs = random_closes()
    p, k = 20, 2
    out = ind.add_bollinger_bands(frame(xs), p, k)
    mid = [None if i + 1 < p else statistics.fmean(xs[i + 1 - p:i + 1]) for i in range(len(xs))]
    sd = [None if i + 1 < p else statistics.stdev(xs[i + 1 - p:i + 1]) for i in range(len(xs))]   # n-1
    assert_matches(out["bb_mid"], mid)
    assert_matches(out["bb_upper"], [None if m is None else m + k * s for m, s in zip(mid, sd)])
    assert_matches(out["bb_lower"], [None if m is None else m - k * s for m, s in zip(mid, sd)])


def test_bollinger_hand_computed():
    out = ind.add_bollinger_bands(frame([1, 2, 3]), 3, 2)
    # mean 2, sample std 1 -> bands 0 and 4
    assert out["bb_mid"].iloc[2] == 2.0
    assert out["bb_upper"].iloc[2] == pytest.approx(4.0)
    assert out["bb_lower"].iloc[2] == pytest.approx(0.0)


def test_bollinger_bands_are_symmetric_and_collapse_on_flat_price():
    out = ind.add_bollinger_bands(frame(random_closes()), 20, 2).dropna()
    np.testing.assert_allclose(out["bb_upper"] - out["bb_mid"], out["bb_mid"] - out["bb_lower"], rtol=1e-12)
    assert (out["bb_upper"] >= out["bb_lower"]).all()
    flat = ind.add_bollinger_bands(frame([9] * 25), 20, 2).dropna()
    assert (flat["bb_upper"] == flat["bb_lower"]).all()


def test_bollinger_wider_multiplier_widens_bands():
    xs = random_closes()
    narrow = ind.add_bollinger_bands(frame(xs), 20, 1)
    wide = ind.add_bollinger_bands(frame(xs), 20, 3)
    assert (wide["bb_upper"].dropna() > narrow["bb_upper"].dropna()).all()


# ---------- volume spike ----------

def test_volume_spike_matches_reference_including_current_candle():
    rng = random.Random(1)
    vols = [rng.uniform(1, 10) for _ in range(150)]
    w, m = 20, 2.0
    out = ind.add_volume_spike(frame([1] * 150, vols), w, m)
    want_avg = [None if i + 1 < w else statistics.fmean(vols[i + 1 - w:i + 1]) for i in range(150)]
    want_spike = [False if a is None else v > a * m for v, a in zip(vols, want_avg)]
    assert_matches(out["volume_avg"], want_avg)
    assert out["volume_spike"].tolist() == want_spike


def test_volume_spike_hand_computed():
    out = ind.add_volume_spike(frame([1] * 4, [1, 1, 1, 10]), 3, 2.0)
    # idx3 average = (1+1+10)/3 = 4 -> threshold 8 -> 10 > 8
    assert out["volume_avg"].iloc[3] == pytest.approx(4.0)
    assert out["volume_spike"].tolist() == [False, False, False, True]


def test_volume_spike_threshold_is_strict_and_warmup_is_false():
    eq = ind.add_volume_spike(frame([1] * 3, [2, 2, 2]), 3, 1.0)
    assert eq["volume_spike"].tolist() == [False, False, False]        # 2 > 2 is False
    warm = ind.add_volume_spike(frame([1] * 5), 3, 2.0)["volume_spike"]
    assert warm.dtype == bool and not warm.any()          # NaN average during warmup -> False, never NaN


def test_volume_spike_current_candle_raises_its_own_bar():
    # A bar 2.5x the previous candles' volume still fails a 2x test, because the
    # average it is compared against includes the bar itself.
    out = ind.add_volume_spike(frame([1] * 4, [1, 1, 1, 2.5]), 3, 2.0)
    # avg = (1+1+2.5)/3 = 1.5 -> threshold 3.0 -> 2.5 does not clear it
    assert out["volume_spike"].iloc[3] == False  # noqa: E712


# ---------- compute_all ----------

def ohlcv(n=120, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close, "volume": rng.uniform(1, 10, n)})


def test_compute_all_adds_every_column_the_signal_layer_reads():
    out = ind.compute_all(ohlcv(), real_cfg)
    # signals.py reads the moving averages under the names config implies (sma_{SMA_FAST} ...);
    # test_signals_price.py checks that a changed period still works end to end.
    for col in [f"sma_{real_cfg.SMA_FAST}", f"sma_{real_cfg.SMA_SLOW}", f"ema_{real_cfg.EMA_FAST}",
                f"ema_{real_cfg.EMA_SLOW}", "rsi", "macd", "macd_signal",
                "bb_upper", "bb_lower", "volume_spike"]:
        assert col in out.columns, f"{col} missing -- signals.py reads it"


def test_compute_all_leaves_source_columns_untouched_and_matches_individual_calls():
    src = ohlcv()
    close, vol = src["close"].copy(), src["volume"].copy()
    out = ind.compute_all(src.copy(), real_cfg)
    pd.testing.assert_series_equal(out["close"], close)
    pd.testing.assert_series_equal(out["volume"], vol)

    xs = list(close)
    assert_matches(out[f"sma_{real_cfg.SMA_FAST}"], ref_sma(xs, real_cfg.SMA_FAST))
    assert_matches(out["rsi"], ref_rsi(xs, real_cfg.RSI_PERIOD))
    assert_matches(out[f"ema_{real_cfg.EMA_SLOW}"], ref_ema(xs, real_cfg.EMA_SLOW))


def test_compute_all_is_idempotent():
    once = ind.compute_all(ohlcv(), real_cfg)
    twice = ind.compute_all(once.copy(), real_cfg)
    pd.testing.assert_frame_equal(once, twice)
