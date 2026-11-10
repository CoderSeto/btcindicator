"""
Tests for kalshi_forecast_study.py. Every piece of math is checked against an
independent reference written here (brute force, numeric integration, a closed
form, or another implementation already in the repo) or against a hand-worked
answer; every feature is checked for look-ahead; and the whole pipeline is run
on a synthetic world whose right answer is known. Nothing touches the network.
(scipy and sklearn are not dependencies of this project, so the references are
brute force rather than library calls.)
"""
import math

import numpy as np
import pandas as pd
import pytest

import config
import indicators
import kalshi
import kalshi_forecast_study as fc
import kalshi_study as ks
import signals

T0 = 1_788_000_000 - (1_788_000_000 % 86400)           # a UTC midnight


# ---------- the family ----------

def test_the_family_is_seven_models_and_fifty_four_intervals():
    assert len(fc.MODELS) == 7 and len(set(fc.MODELS)) == 7
    assert fc.BASELINES == ("market", "asis", "base_rate")
    assert fc.FAMILY_TESTS == 54                                  # (7 x 3 + 6 component pairs) x 2 metrics
    assert len(fc.VOL_KINDS) * len(fc.SPOTS) * len(fc.TAIL_DFS) == 36
    for model, component in fc.COMPONENT_COMPARISONS:
        assert model in ("combiner", "features") and component in fc.MODELS
    assert fc.LOSSES["brier"] is ks.brier and fc.LOSSES["log_loss"] is ks.log_loss      # the edge study's own


# ---------- small numerics ----------

def test_norm_cdf_matches_the_scalar_definition():
    for x in (-6.0, -1.3, 0.0, 0.2, 2.5, 7.0):
        assert float(fc.norm_cdf(x)) == pytest.approx(0.5 * (1 + math.erf(x / math.sqrt(2))), abs=1e-15)


@pytest.mark.parametrize("df", [3, 5])
def test_t_cdf_matches_numeric_integration_of_the_density(df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    grid = np.linspace(-200, 200, 4_000_001)
    pdf = c * (1 + grid ** 2 / df) ** (-(df + 1) / 2)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    cdf += float(fc.t_cdf(-200.0, df))                           # the mass below the grid (~1e-7): negligible either way
    for x in (-8.0, -2.0, -0.5, 0.0, 0.7, 3.0, 10.0):
        assert float(fc.t_cdf(x, df)) == pytest.approx(float(np.interp(x, grid, cdf)), abs=2e-6)
    assert float(fc.t_cdf(0.0, df)) == 0.5
    assert float(fc.t_cdf(1.7, df)) + float(fc.t_cdf(-1.7, df)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fc.t_cdf(0.0, 4)


@pytest.mark.parametrize("df", [3, 5])
def test_the_fat_tail_is_unit_variance_and_fatter_than_the_normal(df):
    x = np.linspace(-400, 400, 2_000_001)
    cdf = fc.tail_probability(x, df)
    var = float(np.sum(((x[1:] + x[:-1]) / 2) ** 2 * np.diff(cdf)))
    # df 5 has unit variance exactly; df 3's variance integral converges too slowly to pin tightly
    assert var == pytest.approx(1.0, abs=0.01 if df == 5 else 0.1)
    assert float(fc.tail_probability(-4.0, df)) > float(fc.tail_probability(-4.0, None))
    assert float(fc.tail_probability(0.0, df)) == 0.5


def test_settlement_z_is_exactly_the_dashboards_model():
    rng = np.random.default_rng(0)
    for _ in range(300):
        spot = float(rng.uniform(20_000, 120_000))
        strike = spot * float(np.exp(rng.normal(0, 0.002)))
        minutes = float(rng.choice([10, 5, 2, 13, 1.5]))
        vol = float(rng.uniform(0.1, 1.5))
        want = kalshi.settlement_probability(spot, strike, minutes, vol)
        assert float(fc.norm_cdf(fc.settlement_z(spot, strike, minutes, vol))) == pytest.approx(want, abs=1e-12)


def test_logit_is_clipped_not_infinite():
    assert np.isfinite(fc.logit([0.0, 1.0])).all()
    assert float(fc.logit(0.5)) == 0.0 and float(fc.logit(0.75)) == pytest.approx(math.log(3))


# ---------- volatility estimators ----------

def brute_ewma(r, half_life):
    w = 0.5 ** (1 / half_life)
    out = []
    for i in range(len(r)):
        num = sum(w ** (i - k) * r[k] ** 2 for k in range(i + 1) if r[k] == r[k])
        den = sum(w ** (i - k) for k in range(i + 1) if r[k] == r[k])
        out.append(num / den if den > 0 else float("nan"))
    return np.array(out)


def test_ewma_variance_matches_brute_force_with_missing_minutes():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 3e-4, 300)
    r[[0, 40, 41, 150]] = np.nan
    got, want = fc.ewma_variance(r, 10), brute_ewma(r, 10)
    assert math.isnan(got[0]) and np.allclose(got[1:], want[1:], rtol=1e-10)


def test_ewma_half_life_means_what_it_says():
    # one unit return 10 minutes ago followed by zeros: its weight is exactly half the newest one's
    r = np.zeros(11)
    r[0] = 1.0
    w = 0.5 ** (np.arange(11) / 10)
    assert fc.ewma_variance(r, 10)[-1] == pytest.approx(0.5 / w.sum())


def brute_bipower(r, window, min_pairs):
    out = np.full(len(r), np.nan)
    for i in range(len(r)):
        prods = [abs(r[k]) * abs(r[k - 1]) for k in range(max(1, i - window + 1), i + 1)
                 if r[k] == r[k] and r[k - 1] == r[k - 1]]
        # the rolling window holds `window` slots; it needs min_pairs of them filled
        if len(prods) >= min_pairs:
            out[i] = math.pi / 2 * sum(prods) / len(prods)
    return out


def test_bipower_variance_matches_brute_force():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 3e-4, 400)
    r[[0, 77, 200, 201]] = np.nan
    got, want = fc.bipower_variance(r, 120, 60), brute_bipower(r, 120, 60)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-9)


def test_bipower_estimates_the_variance_and_shrugs_off_a_jump_that_ewma_does_not():
    rng = np.random.default_rng(3)
    sd = 3e-4
    r = rng.normal(0, sd, 5000)
    assert np.nanmean(fc.bipower_variance(r)[200:]) == pytest.approx(sd ** 2, rel=0.05)
    jump = r.copy()
    jump[4990] = 50 * sd                                          # one 50-sigma minute
    bp_ratio = fc.bipower_variance(jump)[-1] / fc.bipower_variance(r)[-1]
    ew_ratio = fc.ewma_variance(jump, 60)[-1] / fc.ewma_variance(r, 60)[-1]
    # the jump enters bipower as two products with ordinary neighbours (~80 sd^2 in 120 pairs), EWMA as 2500 sd^2
    assert bp_ratio < 4.0 < 10.0 < ew_ratio


def test_no_volatility_estimate_reads_a_later_return():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 3e-4, 400)
    later = r.copy()
    later[300:] *= 10
    for f in (lambda x: fc.ewma_variance(x, 30), fc.bipower_variance):
        assert np.array_equal(f(r)[:300], f(later)[:300], equal_nan=True)
        assert f(r)[300] != f(later)[300]


def test_seasonal_factors_by_hand_and_only_from_before_the_cutoff():
    # two days of minutes; hour 0 returns are +-2e-4, every other hour +-1e-4
    ts = T0 + 60 * np.arange(2880)
    hour = (ts // 3600) % 24
    r = np.where(hour == 0, 2e-4, 1e-4) * np.where(np.arange(2880) % 2, 1, -1)
    f = fc.seasonal_factors(ts, r, T0 + 2 * 86400)
    overall = (4 + 23) / 24                                       # mean r^2 in units of 1e-8
    assert f[0] == pytest.approx(math.sqrt(4 / overall)) and f[5] == pytest.approx(math.sqrt(1 / overall))
    # a wild second day is invisible when the cutoff is the end of day one
    wild = r.copy()
    wild[1440:] *= 100
    assert np.allclose(fc.seasonal_factors(ts, wild, T0 + 86400), fc.seasonal_factors(ts, r, T0 + 86400))
    # an hour with no data before the cutoff gets a neutral factor
    assert fc.seasonal_factors(ts, r, T0 + 3600)[7] == 1.0


def grid_of(closes, t_start=T0):
    closes = np.asarray(closes, dtype=float)
    ts = t_start + 60 * np.arange(len(closes))
    return ks.MinuteGrid(pd.DataFrame({"ts": ts, "open": closes, "high": closes, "low": closes, "close": closes}))


def test_vol_table_recovers_a_known_volatility_in_every_estimator():
    sd = 3e-4
    closes = 100_000 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, sd, 6000)))
    table = fc.vol_table(grid_of(closes), T0 + 60 * 3000)
    true = sd * math.sqrt(fc.MINUTES_PER_YEAR)
    assert set(table) == set(fc.VOL_KINDS) - {"rv120"}
    for kind, v in table.items():
        assert len(v) == 6000
        assert float(np.nanmean(v[500:])) == pytest.approx(true, rel=0.06), kind
    # entry i uses candle i and earlier: the dashboard's realized vol at the same point agrees with ewma loosely,
    # and exactly matches a numpy calculation on the same 121 closes
    i = 4000
    rv = ks.model_inputs(grid_of(closes), T0 + 60 * (i + 1))[1]
    assert rv == pytest.approx(float(np.std(np.diff(np.log(closes[i - 120:i + 1])), ddof=1)) * math.sqrt(fc.MINUTES_PER_YEAR))
    assert table["ewma60"][i] == pytest.approx(rv, rel=0.25)


# ---------- the bot's read ----------

def candle_frame(n, seed=0, t_start=T0, sd=4e-4):
    rng = np.random.default_rng(seed)
    close = 75_000 * np.exp(np.cumsum(rng.normal(0, sd, n)))
    open_ = np.concatenate([[75_000.0], close[:-1]])
    return pd.DataFrame({"ts": t_start + 60 * np.arange(n), "open": open_, "high": np.maximum(open_, close) * 1.0002,
                         "low": np.minimum(open_, close) * 0.9998, "close": close,
                         "volume": rng.uniform(0.5, 6.0, n)})


def test_bot_votes_are_what_the_bot_would_have_computed_with_only_the_candles_closed_by_then():
    cb = candle_frame(400, seed=6)
    quote_ts = np.array([T0 + 60 * k for k in (120, 233, 399, 400)])
    got = fc.bot_net_votes(cb, quote_ts)
    for t, g in zip(quote_ts, got):
        seen = cb[cb.ts <= t - 60]                               # everything closed by the quote minute, nothing later
        snap = signals.evaluate_price_only(indicators.compute_all(seen.reset_index(drop=True), config), config)
        assert g == snap.bullish_count - snap.bearish_count
    assert set(np.unique(got)) <= set(range(-6, 7)) and len(set(got)) > 1


def test_bot_votes_are_missing_without_the_candle_or_before_warm_up():
    cb = candle_frame(200, seed=7)
    holed = cb[cb.ts != T0 + 60 * 150]
    got = fc.bot_net_votes(holed, np.array([T0 + 60 * 151, T0 + 60 * 5, T0 + 60 * 500]))
    assert np.isnan(got).all()                                   # candle missing; not warmed up; beyond the data


# ---------- logistic regression ----------

def objective(model, X, y, l2):
    Z = np.column_stack([np.ones(len(y)), (X - model["mean"]) / model["sd"]])
    eta = Z @ model["beta"]
    return float(np.sum(np.log1p(np.exp(eta)) - y * eta) + l2 / 2 * np.sum(model["beta"][1:] ** 2))


def test_logistic_on_one_binary_input_reproduces_the_two_group_rates():
    # group x=0: 30 of 100 are YES; group x=1: 80 of 100. The unpenalized MLE fits both exactly.
    x = np.array([0] * 100 + [1] * 100, dtype=float)
    y = np.array([1] * 30 + [0] * 70 + [1] * 80 + [0] * 20, dtype=float)
    m = fc.fit_logistic(x[:, None], y)
    p = 1 / (1 + np.exp(-(np.column_stack([np.ones(2), (np.array([[0.0], [1.0]]) - m["mean"]) / m["sd"]]) @ m["beta"])))
    assert p == pytest.approx([0.30, 0.80], abs=1e-8)
    assert fc.predict_logistic(m, np.array([[0.0], [1.0]])) == pytest.approx([0.30, 0.80], abs=1e-8)


def test_platt_scaling_matches_the_edge_studys_own_calibration_fit():
    rng = np.random.default_rng(8)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.2 + 1.4 * fc.logit(p))))).astype(float)
    a, b = ks.calibration_fit(p, y)                              # an independent Newton fit, unstandardized
    m = fc.fit_logistic(fc.logit(p)[:, None], y)
    slope = m["beta"][1] / m["sd"][0]
    intercept = m["beta"][0] - slope * m["mean"][0]
    assert (intercept, slope) == pytest.approx((a, b), abs=1e-6)


@pytest.mark.parametrize("l2", [0.0, 1.0, 50.0])
def test_logistic_solution_is_the_minimum_of_the_penalized_objective(l2):
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, (600, 4)) * [1, 5, 0.1, 2]
    y = (rng.random(600) < 1 / (1 + np.exp(-(0.3 + X @ [0.8, -0.1, 3.0, 0.0])))).astype(float)
    m = fc.fit_logistic(X, y, l2)
    best = objective(m, X, y, l2)
    for _ in range(200):                                         # brute force: no nearby point is lower
        other = {**m, "beta": m["beta"] + rng.normal(0, 0.05, 5)}
        assert objective(other, X, y, l2) >= best - 1e-9
    # numeric gradient is zero
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-5
        g = (objective({**m, "beta": m["beta"] + e}, X, y, l2) - objective({**m, "beta": m["beta"] - e}, X, y, l2)) / 2e-5
        assert abs(g) < 1e-4


def test_l2_shrinks_the_slopes_but_not_the_intercept_and_a_constant_column_is_harmless():
    rng = np.random.default_rng(10)
    X = rng.normal(0, 1, (500, 2))
    y = (rng.random(500) < 1 / (1 + np.exp(-(1.0 + 2 * X[:, 0])))).astype(float)
    free, tight = fc.fit_logistic(X, y, 0.0), fc.fit_logistic(X, y, 1e6)
    assert abs(tight["beta"][1]) < 0.01 < abs(free["beta"][1])
    assert tight["beta"][0] == pytest.approx(math.log(y.mean() / (1 - y.mean())), abs=1e-3)      # falls back to the base rate
    const = fc.fit_logistic(np.column_stack([X, np.ones(500)]), y, 1.0)
    assert np.isfinite(const["beta"]).all() and abs(const["beta"][3]) < 1e-9


def test_fitted_probabilities_are_clipped_away_from_zero_and_one():
    m = fc.fit_logistic(np.array([[0.0], [1.0], [0.0], [1.0]]), np.array([0.0, 1.0, 0.0, 1.0]), 0.01)
    p = fc.predict_logistic(m, np.array([[-50.0], [50.0]]))
    assert p[0] == fc.PROB_CLIP and p[1] == 1 - fc.PROB_CLIP


# ---------- isotonic regression ----------

def brute_isotonic(x, y):
    """The textbook max-min formula on the tie-pooled points: fit_k = max_{a<=k} min_{b>=k} weighted mean(a..b)."""
    ux = np.unique(x)
    w = np.array([np.sum(x == u) for u in ux], dtype=float)
    m = np.array([np.mean(y[x == u]) for u in ux])
    n = len(ux)
    out = np.empty(n)
    for k in range(n):
        out[k] = max(min(np.sum(w[a:b + 1] * m[a:b + 1]) / np.sum(w[a:b + 1]) for b in range(k, n)) for a in range(k + 1))
    return ux, out


def test_isotonic_matches_the_max_min_formula_with_ties():
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(2, 30))
        x = np.round(rng.uniform(0, 1, n), 1)                    # rounding makes ties
        y = (rng.random(n) < x).astype(float)
        m = fc.fit_isotonic(x, y)
        ux, want = brute_isotonic(x, y)
        assert np.array_equal(m["x"], ux) and np.allclose(m["y"], want)
        assert (np.diff(m["y"]) >= -1e-12).all()


def test_isotonic_by_hand_and_its_prediction():
    # y = 0, 1, 0, 1 at x = 1..4: the middle pair violates and pools to 0.5
    m = fc.fit_isotonic([1, 2, 3, 4], [0, 1, 0, 1])
    assert list(m["y"]) == [0.0, 0.5, 0.5, 1.0]
    p = fc.predict_isotonic(m, [0.0, 1.0, 1.5, 2.5, 4.0, 9.0])
    assert p == pytest.approx([fc.PROB_CLIP, fc.PROB_CLIP, 0.25, 0.5, 1 - fc.PROB_CLIP, 1 - fc.PROB_CLIP])


# ---------- skill, reliability, scorecard ----------

def test_skill_by_hand():
    # model loss mean 0.1, reference 0.2: the model removes half the loss
    lm = np.array([0.1, 0.1, 0.1, 0.1])
    lr = np.array([0.2, 0.2, 0.2, 0.2])
    s = fc.skill(lm, lr, ["a", "a", "b", "b"], z=2.0)
    assert s["skill"] == pytest.approx(0.5) and s["se"] == pytest.approx(0.0) and s["verdict"] == "BETTER"
    assert fc.skill(lr, lm, ["a", "a", "b", "b"], z=2.0)["skill"] == pytest.approx(-1.0)
    assert fc.skill(lr, lm, ["a", "a", "b", "b"], z=2.0)["verdict"] == "WORSE"
    assert fc.skill(lm, lm, ["a", "a", "b", "b"], z=2.0)["verdict"] == "not distinguishable"      # zero is not above zero


def test_skill_interval_matches_a_brute_force_bootstrap_of_the_ratio_itself():
    rng = np.random.default_rng(12)
    n_days, per = 80, 30
    days = np.repeat(np.arange(n_days), per)
    day_effect = np.repeat(rng.normal(0, 0.03, n_days), per)                   # same-day rows move together
    lr = np.clip(0.2 + day_effect + rng.normal(0, 0.1, n_days * per), 0.001, None)
    lm = np.clip(lr - 0.01 + rng.normal(0, 0.05, n_days * per), 0.001, None)
    got = fc.skill(lm, lr, days, z=1.0)
    idx = [np.flatnonzero(days == d) for d in range(n_days)]
    draws = []
    for _ in range(1500):                                                      # resample whole days, recompute the RATIO
        pick = np.concatenate([idx[d] for d in rng.integers(0, n_days, n_days)])
        draws.append(1 - lm[pick].mean() / lr[pick].mean())
    assert got["skill"] == pytest.approx(1 - lm.mean() / lr.mean())
    assert got["se"] == pytest.approx(float(np.std(draws, ddof=1)), rel=0.10)
    assert got["hi"] - got["lo"] == pytest.approx(2 * got["se"])
    assert got["n_eff"] < got["n"]                                             # same-day correlation costs sample size


def test_skill_is_not_fooled_by_many_rows_on_few_days():
    rng = np.random.default_rng(13)
    days = np.repeat(np.arange(4), 2000)                                       # 8,000 rows, 4 independent days
    shift = np.repeat(np.array([0.05, -0.04, 0.03, -0.045]), 2000)
    lr = 0.2 + rng.normal(0, 0.01, 8000)
    s = fc.skill(lr - shift, lr, days, z=ks.bonferroni_z(fc.FAMILY_TESTS))
    assert s["verdict"] == "not distinguishable" and s["n_eff"] < 50


def small_predictions():
    """Hand-built held-out predictions: 3 days, forecasts chosen so every number below can be worked by hand."""
    rows = []
    for day, (m, a, y) in enumerate([(0.9, 0.8, 1), (0.9, 0.6, 1), (0.1, 0.3, 0)]):
        for k in range(4):
            rows.append({"ticker": f"T{day}{k}", "close_ts": T0 + 86400 * day + 900 * k, "day": f"d{day}", "horizon": 10,
                         "outcome": y if k < 3 else 1 - y, "market": m, "asis": a, "base_rate": 0.5})
    p = pd.DataFrame(rows)
    for name in fc.MODELS:
        p[name] = p["market"]
    return p


def test_scorecard_matches_plain_python_loops():
    p = small_predictions()
    card = fc.scorecard(p, z=2.0, calibration_resamples=20).set_index("forecast")
    y = p.outcome.to_numpy()
    for name in ("market", "asis", "base_rate"):
        f = p[name].to_numpy()
        brier = sum((fi - yi) ** 2 for fi, yi in zip(f, y)) / len(y)
        ll = -sum(math.log(fi) if yi else math.log(1 - fi) for fi, yi in zip(f, y)) / len(y)
        assert card.loc[name, "brier"] == pytest.approx(brier) and card.loc[name, "log_loss"] == pytest.approx(ll)
    # market: forecasts 0.9 x8 and 0.1 x4 -> mean 0.6333, sd = 0.8 * sqrt(2/9); |p - 0.5| is 0.4 everywhere
    assert card.loc["market", "sharp_abs"] == pytest.approx(0.4)
    assert card.loc["market", "sharp_sd"] == pytest.approx(0.8 * math.sqrt(2 / 9))
    # the base rate says 0.5 every time: zero sharpness, Brier 0.25, and no calibration slope to report
    assert card.loc["base_rate", "sharp_sd"] == 0.0 and card.loc["base_rate", "brier"] == pytest.approx(0.25)
    assert math.isnan(card.loc["base_rate", "cal_b"])
    # the slope is the edge study's own fit; with three days a resample can be constant, so the SE may be absent
    assert card.loc["asis", "cal_b"] == pytest.approx(ks.calibration_fit(p.asis, y)[1])
    assert list(card.index[:3]) == list(fc.BASELINES) and list(card["kind"][:3]) == ["baseline"] * 3


def test_a_constant_forecast_is_perfectly_calibrated_and_has_no_skill():
    # the point the output makes in words: calibration alone is trivial
    rng = np.random.default_rng(14)
    y = (rng.random(4000) < 0.5).astype(float)
    days = np.repeat(np.arange(40), 100)
    t = fc.reliability_table(np.full(4000, y.mean()), y, days, z=2.0)
    assert len(t) == 1 and t.forecast.iloc[0] == pytest.approx(t.yes_rate.iloc[0])               # "reliable"
    informed = np.where(y == 1, 0.8, 0.2)
    s = fc.skill(ks.brier(informed, y), ks.brier(np.full(4000, y.mean()), y), days, z=3.0)
    assert s["verdict"] == "BETTER" and s["skill"] > 0.8


def test_reliability_table_by_hand():
    # bucket 0.1-0.2: forecasts 0.15 on two days, 1 YES of 4. bucket 0.8-0.9: forecasts 0.85, 3 YES of 4.
    p = np.array([0.15] * 4 + [0.85] * 4)
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=float)
    days = np.array(["a", "a", "b", "b"] * 2)
    t = fc.reliability_table(p, y, days, z=2.0).set_index("bucket")
    assert list(t.index) == ["0.1-0.2", "0.8-0.9"]
    assert t.loc["0.1-0.2", "n"] == 4 and t.loc["0.1-0.2", "forecast"] == pytest.approx(0.15)
    assert t.loc["0.1-0.2", "yes_rate"] == pytest.approx(0.25) and t.loc["0.8-0.9", "yes_rate"] == pytest.approx(0.75)
    assert t.small.all()                                         # four rows are flagged, not presented at face value
    assert (t.lo >= 0).all() and (t.hi <= 1).all() and (t.lo <= t.yes_rate).all() and (t.yes_rate <= t.hi).all()


def test_reliability_interval_is_the_edge_studys_day_block_bootstrap():
    rng = np.random.default_rng(15)
    p = rng.uniform(0.31, 0.39, 3000)
    y = (rng.random(3000) < 0.35).astype(float)
    days = np.repeat(np.arange(60), 50)
    t = fc.reliability_table(p, y, days, z=2.5).iloc[0]
    b = ks.block_bootstrap(y, days)
    assert (t.lo, t.hi) == pytest.approx(ks.interval(b, 2.5)) and t.n_eff == pytest.approx(b["n_eff"]) and not t.small


def test_skill_table_has_every_counted_interval_once():
    p = small_predictions()
    p["platt_asis"] = p["asis"]
    t = fc.skill_table(p, z=2.0)
    assert len(t) == fc.FAMILY_TESTS
    assert not t.duplicated(["model", "reference", "metric"]).any()
    assert (t[t.kind == "baseline"].groupby("model").size() == 6).all()
    # a model identical to its reference has exactly zero skill and is never called better
    same = t[(t.model == "platt_market") & (t.reference == "market")]
    assert (same.skill.abs() < 1e-12).all() and (same.verdict == "not distinguishable").all()


def test_by_horizon_is_the_plain_mean_per_horizon():
    p = small_predictions()
    p.loc[p.index[:4], "horizon"] = 5
    t = fc.by_horizon(p)
    g = p[p.horizon == 5]
    want = float(((g.market - g.outcome) ** 2).mean())
    assert float(t[(t.horizon == 5) & (t.forecast == "market")].brier.iloc[0]) == pytest.approx(want)
    assert len(t) == 2 * (len(fc.BASELINES) + len(fc.MODELS))


# ---------- the whole pipeline on a synthetic world with a known answer ----------

SD = 3e-4
TRUE_VOL = SD * math.sqrt(fc.MINUTES_PER_YEAR)


def synthetic_world(n_days=36, per_day=40, seed=0, market="shrunk", t0=T0):
    """
    Coinbase is a constant-volatility random walk. Contracts run back to back from 06:00 UTC;
    each strike is the last close before its open and it settles YES iff the last close before
    its close is at or above the strike. The TRUE probability at a quote minute is therefore
    known (a normal CDF over the minutes left). The market mid is either that truth ("true")
    or the truth pulled halfway to 50% ("shrunk": a miscalibrated market that Platt scaling of
    the mid can fix, and nothing else is needed to fix).
    """
    cb = candle_frame(1440 * n_days, seed=seed, t_start=t0, sd=SD)
    close = cb["close"].to_numpy()
    markets, candles = [], {}
    for d in range(n_days):
        for j in range(per_day):
            open_ts = t0 + 86400 * d + 6 * 3600 + 900 * j
            i_open = (open_ts - t0) // 60
            strike = float(close[i_open - 1])
            outcome = int(close[i_open + 14] >= strike)
            tk = f"W{d:03d}-{j:02d}"
            q = {}
            for k in range(1, 15):                               # quote at open + k minutes: 15 - k minutes left
                spot = float(close[i_open + k - 1])
                truth = float(fc.norm_cdf(math.log(spot / strike) / (SD * math.sqrt(15 - k))))
                mid = truth if market == "true" else 0.5 + 0.5 * (truth - 0.5)
                mid = min(max(mid, 0.03), 0.97)
                q[open_ts + 60 * k] = (round(mid - 0.01, 4), round(mid + 0.01, 4), mid)
            markets.append({"ticker": tk, "open_ts": open_ts, "close_ts": open_ts + 900, "strike": strike,
                            "outcome": outcome, "expiration_value": float(close[i_open + 14])})
            candles[tk] = q
    return markets, candles, cb


@pytest.fixture(scope="module")
def world():
    markets, candles, cb = synthetic_world()
    d = fc.build_dataset(markets, candles, ks.MinuteGrid(cb), cb)
    fitted = fc.tune_and_fit(d)
    pred = fc.predictions(fitted, d[d.usable & (d.split == "test")])
    return {"markets": markets, "candles": candles, "cb": cb, "d": d, "fitted": fitted, "pred": pred}


def test_dataset_has_one_row_per_contract_and_horizon_split_in_time_order(world):
    d, markets = world["d"], world["markets"]
    assert len(d) == 3 * len(markets) and not d.duplicated(["ticker", "horizon"]).any()
    val_cut, test_cut = fc.split_cutoffs(markets)
    assert d[d.split == "fit"].close_ts.max() < val_cut <= d[d.split == "validation"].close_ts.min()
    assert d[d.split == "validation"].close_ts.max() < test_cut <= d[d.split == "test"].close_ts.min()
    # the held-out cutoff is the edge study's own, and validation is the last quarter of train
    assert test_cut == ks.split_cutoff([m["close_ts"] for m in markets])
    n = d.groupby("split").ticker.nunique()
    assert n["test"] == pytest.approx(0.30 * len(markets), abs=2)
    assert n["validation"] == pytest.approx(0.25 * (len(markets) - n["test"]), abs=2)
    assert d.usable.mean() > 0.95


def test_the_as_is_column_is_the_dashboards_model_on_the_last_closed_candle(world):
    d, cb = world["d"], world["cb"]
    r = d[d.usable & (d.horizon == 5)].iloc[50]
    close = cb.set_index("ts")["close"]
    spot = float(close[r.quote_ts - 60])
    window = close[(close.index <= r.quote_ts - 60) & (close.index >= r.quote_ts - 60 - 120 * 60)].to_numpy()
    vol = float(np.std(np.diff(np.log(window)), ddof=1)) * math.sqrt(fc.MINUTES_PER_YEAR)
    assert r.p_asis == pytest.approx(kalshi.settlement_probability(spot, r.strike, 5, vol), abs=1e-9)
    assert float(fc.norm_cdf(r[fc.z_column("rv120", "coinbase")])) == pytest.approx(r.p_asis, abs=1e-9)
    # momentum: the 5-minute log return in units of the realized per-minute sd
    want = math.log(spot / float(close[r.quote_ts - 60 - 300])) / (vol / math.sqrt(fc.MINUTES_PER_YEAR) * math.sqrt(5))
    assert r.mom5 == pytest.approx(want, rel=1e-9)


def test_no_feature_of_a_row_changes_when_everything_from_its_quote_minute_on_changes(world):
    markets, candles, cb = world["markets"], world["candles"], world["cb"]
    # a held-out contract: the seasonality table is fitted on minutes before the validation cutoff (a train-side
    # parameter, like any coefficient), so only rows after that cutoff are free of everything later than themselves
    target = markets[int(len(markets) * 0.85)]
    quote_ts = target["close_ts"] - 600                          # its 10-minute row
    cb2 = cb.copy()
    later = cb2.ts >= quote_ts                                   # the candle starting at the quote minute, and all after
    cb2.loc[later, ["open", "high", "low", "close"]] *= 1.03
    cb2.loc[later, "volume"] *= 7
    markets2 = [dict(m, outcome=1 - m["outcome"]) if m["close_ts"] > target["open_ts"] else m for m in markets]
    candles2 = {t: {ts: ((0.2, 0.22, 0.21) if ts > quote_ts else v) for ts, v in q.items()} for t, q in candles.items()}
    d2 = fc.build_dataset(markets2, candles2, ks.MinuteGrid(cb2), cb2)
    cols = ["p_asis", "mid", "mom5", "log_vol_ratio", "basis_dev", "streak", "bot_net"] + \
           [fc.z_column(k, s) for k in fc.VOL_KINDS for s in fc.SPOTS]
    a = world["d"][(world["d"].ticker == target["ticker"]) & (world["d"].horizon == 10)].iloc[0]
    b = d2[(d2.ticker == target["ticker"]) & (d2.horizon == 10)].iloc[0]
    for c in cols:
        assert a[c] == pytest.approx(b[c], rel=1e-12, abs=1e-12), c
    # ...and the check has teeth: the candle just before the quote minute IS an input
    cb3 = cb.copy()
    cb3.loc[cb3.ts == quote_ts - 60, "close"] *= 1.001
    d3 = fc.build_dataset(markets, candles, ks.MinuteGrid(cb3), cb3)
    c = d3[(d3.ticker == target["ticker"]) & (d3.horizon == 10)].iloc[0]
    assert c.p_asis != a.p_asis and c.mom5 != a.mom5


def test_tuning_and_fitting_never_read_the_held_out_rows(world):
    d = world["d"].copy()
    test = d.split == "test"
    d.loc[test, "outcome"] = 1 - d.loc[test, "outcome"]
    d.loc[test, ["mid", "p_asis", "mom5", "bot_net"]] = 0.123
    again = fc.tune_and_fit(d)
    first = world["fitted"]
    assert again["structural_choice"] == first["structural_choice"] and again["l2"] == first["l2"]
    assert again["base_rate"] == first["base_rate"]
    assert again["tuning"].equals(first["tuning"])
    for m in fc.MODELS:
        for h in ks.HORIZONS_MIN:
            a, b = first["models"][m]["per_horizon"][h], again["models"][m]["per_horizon"][h]
            key = "y" if m == "iso_asis" else "beta"
            assert np.array_equal(a[key], b[key])


def test_tuning_covers_every_candidate_and_picks_the_lowest_validation_loss(world):
    f = world["fitted"]
    t = f["tuning"]
    assert (t.model == "structural").sum() == 36 and (t.model == "features").sum() == 4 and (t.model == "combiner").sum() == 4
    kind, spot, df = f["structural_choice"]
    best = t[t.model == "structural"].sort_values("val_log_loss").iloc[0]
    assert best.setting == f"{kind} / {spot} / tail {df or 'normal'}"
    for model in ("features", "combiner"):
        assert f"l2 {f['l2'][model]:g}" == t[t.model == model].sort_values("val_log_loss").iloc[0].setting
    # constant true volatility, so a fat tail has nothing to fix: the normal is chosen
    assert df is None


def test_predictions_are_side_by_side_probabilities_for_every_forecast(world):
    pred = world["pred"]
    for name in fc.BASELINES + fc.MODELS:
        assert pred[name].between(0, 1).all() and pred[name].notna().all()
    for name in fc.MODELS:
        assert pred[name].between(fc.PROB_CLIP, 1 - fc.PROB_CLIP).all()
    assert pred["base_rate"].nunique() == 1
    assert pred["base_rate"].iloc[0] == pytest.approx(world["fitted"]["base_rate"])


def test_a_miscalibrated_market_is_fixed_by_recalibrating_it_and_the_verdicts_say_so(world):
    pred = world["pred"]
    z = ks.bonferroni_z(fc.FAMILY_TESTS)
    t = fc.skill_table(pred, z)

    def verdict(model, ref, metric="brier"):
        return t[(t.model == model) & (t.reference == ref) & (t.metric == metric)].verdict.iloc[0]

    assert verdict("platt_market", "market") == "BETTER" and verdict("platt_market", "market", "log_loss") == "BETTER"
    assert verdict("combiner", "market") == "BETTER"
    assert verdict("bot_only", "market") == "WORSE"              # a vote count is no match for a price
    for m in fc.MODELS:
        if m != "bot_only":
            assert verdict(m, "base_rate") == "BETTER", m
    # the as-is model is already right in this world: recalibrating it cannot beat it
    assert verdict("platt_asis", "asis") == "not distinguishable"
    card = fc.scorecard(pred, z, calibration_resamples=30).set_index("forecast")
    assert card.loc["market", "cal_b"] > 1.5                     # prices too close to 50%: slope well above 1
    assert card.loc["platt_market", "cal_b"] == pytest.approx(1.0, abs=0.15)
    assert card.loc["platt_market", "sharp_sd"] > card.loc["market", "sharp_sd"]
    assert card.loc["platt_market", "brier"] < card.loc["market", "brier"]
    # the fitted base rate is a constant whose sd is float residue, not exactly 0: still no slope is reported for it
    assert math.isnan(card.loc["base_rate", "cal_b"]) and card.loc["base_rate", "sharp_sd"] < 1e-9


def test_nothing_is_called_better_than_a_market_that_is_already_right():
    markets, candles, cb = synthetic_world(n_days=30, per_day=32, seed=4, market="true")
    d = fc.build_dataset(markets, candles, ks.MinuteGrid(cb), cb)
    pred = fc.predictions(fc.tune_and_fit(d), d[d.usable & (d.split == "test")])
    t = fc.skill_table(pred, ks.bonferroni_z(fc.FAMILY_TESTS))
    vs_market = t[t.reference == "market"]
    assert not (vs_market.verdict == "BETTER").any()
    assert (vs_market[vs_market.model == "bot_only"].verdict == "WORSE").all()
