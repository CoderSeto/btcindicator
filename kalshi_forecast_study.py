"""
Can P(YES) for a KXBTC15M contract be forecast better than the baselines?
A forecasting study, NOT a trading rule. Reusable study code.

No fee, fill or profit appears anywhere here. The question is only how good a
probability is: scored by Brier and log loss on held-out contracts, against
three baselines -- Kalshi's own mid, kalshi.settlement_probability exactly as
the dashboard computes it, and the base rate.

Two things to hold in mind while reading any number this produces, both
printed in the output too:
  - The market mid is a STRONG baseline. The edge study found it better than
    the dashboard's model at every horizon. Matching it is not a failure.
  - Calibration alone is trivial: "always say the base rate" is perfectly
    calibrated and useless. A forecast needs calibration AND sharpness, which
    is why Brier / log loss decide and the reliability table only describes.

This module holds the features, the models and the metrics, all tested on
hand-built data. `review/kalshi_forecast_backtest.py` runs it on the cache.

PRE-REGISTERED DESIGN. Fixed, and committed, before any model here had been
fitted to or scored on any outcome. Changing anything after seeing held-out
results is tuning on them; add a model instead, count it in the family, and
label it as an amendment below.

What the designer HAD seen, stated because it is the honest limit of "blind":
REVIEW_FINDINGS.md, which prints held-out Brier / log loss for the mid and for
settlement_probability, their calibration slopes, and descriptive held-out
results of the trading studies (including the four-hypothesis study the same
designer ran on this cache). So the held-out contracts are not untouched: the
baselines' scores on them were known, and so was the direction of several
small effects. No model below had been fitted. Nothing newer exists to use
instead (see Split).

  - Data: the study_data/ cache as it stands (30,950 settled contracts closing
    2025-12-15 .. 2026-11-10, their 1-minute Kalshi quotes, Coinbase 1-minute
    candles). Not refetched. Observations are kalshi_study.observations: one
    row per contract per horizon in ks.HORIZONS_MIN (10, 5, 2 minutes before
    close), the quote standing at exactly that minute.
  - Split, chronological, by contract close time, using ks.split_cutoff:
      held-out   the last ks.TEST_FRACTION (30%) of contracts: the EXISTING
                 held-out set, closing from 2026-08-04. The task asked to
                 prefer contracts after that cutoff if any exist. They do, and
                 they are exactly this set (through 2026-11-10); the cache
                 holds nothing later, and a few hours of newer contracts
                 would be far too few. So this set is used, and it is NOT
                 pristine (see above).
      train      everything before it, itself split by ks.split_cutoff with
                 VALIDATION_FRACTION (25%): `fit` (the earlier 75% of train
                 contracts) and `validation` (the later 25%).
    Tuning: every candidate is fitted on `fit` and compared on `validation` by
    pooled log loss. The chosen settings are then refitted on all of train.
    The held-out set is scored ONCE, after every choice is final.
  - No look-ahead. A row's features use only: its own quote at the quote
    minute; Coinbase candles that had CLOSED by then (the newest is the one
    covering [t - 60, t); the runner also drops a forming candle with
    data_fetch.drop_forming_candle); the strike (published at open); results
    of contracts that closed at or before this one opened. Coinbase history
    is the ranged /candles cache (gotcha 19). Fitted parameters and the
    seasonality table come from train rows / train-period minutes only.
  - Rows scored: `usable` rows only -- a real book (ks.valid_book), a strike
    within kalshi.MAX_STRIKE_DISTANCE of spot, and every feature present.
    Every model and every baseline is scored on the SAME rows. The count
    dropped is printed. (At 2 minutes ~15% of contracts have no two-sided
    book: the most decided ones are missing from every score.)
  - Features (FEATURES):
      p_asis        ks.model_observations' p_model: settlement_probability on
                    the last Coinbase trade and the 120-minute realized vol.
      z[kind,spot]  the model's z-score, (ln(S/K) - v/2) / sqrt(v) with
                    v = sigma^2 * kalshi.settlement_variance_seconds / year,
                    for each volatility estimator in VOL_KINDS and each spot in
                    SPOTS ("coinbase"; "basis_adjusted" = spot * strike / the
                    Coinbase OHLC mean of the minute before open, as ks does).
      VOL_KINDS     rv120 (the dashboard's, as-is); ewma10 / ewma30 / ewma60
                    (EWMA of squared 1-minute log returns, half-life in
                    minutes, decaying across missing minutes); bipower120
                    (jump-robust: (pi/2) * mean |r_i||r_{i-1}| over 120
                    minutes); ewma30_seasonal (ewma30 of returns divided by an
                    hour-of-day factor, times the factor of the quote's hour;
                    the 24 factors are sqrt(mean r^2 in that UTC hour / mean
                    r^2 overall) over minutes before the validation cutoff).
      mom5          5-minute Coinbase log return / (rv120 per-minute sd * sqrt 5).
      log_vol_ratio ln(ewma10 vol / rv120 vol).
      basis_dev     kalshi_strategy_study.basis_features (bp; missing -> 0).
      streak        kalshi_strategy_study.streak_features (missing -> 0).
      bot_net       the bot's own read: bullish minus bearish votes of
                    signals.evaluate_price_only on the candles closed by the
                    quote minute, indicators.compute_all with config.py's
                    settings (-6 .. +6). Its "confidence" is a direction
                    count, not a probability, so it enters only as a model
                    input and as the fitted `bot_only` model.
      mid           the Kalshi mid (the market models and the combiner only).
  - Model class: logistic regression fitted by Newton's method with an L2
    penalty on standardized inputs (intercept unpenalized), one fit per
    horizon; plus one isotonic regression. Probabilities enter as logits
    clipped to [LOGIT_EPS, 1 - LOGIT_EPS]. Every fitted model's output is
    clipped to [PROB_CLIP, 1 - PROB_CLIP]. The baselines are scored as they
    are (ks.log_loss clips at its own 1e-4).
  - The family: 7 models (MODELS).
      platt_asis    logit(p_asis) -> Platt scaling.
      iso_asis      isotonic regression of the outcome on p_asis.
      structural    one (vol kind, spot, tail) chosen on validation from
                    VOL_KINDS x SPOTS x TAIL_DFS (6 x 2 x 3 = 36 candidates),
                    then Platt-scaled. Tail: the z-score through a normal CDF
                    (df None) or a unit-variance Student-t CDF with 5 or 3
                    degrees of freedom (fat tails).
      features      structural's logit + mom5, log_vol_ratio, basis_dev,
                    streak, bot_net. L2 strength chosen on validation from
                    L2_GRID. No market input.
      bot_only      bot_net alone.
      platt_market  logit(mid) -> Platt scaling.
      combiner      logit(mid), structural's logit, logit(p_asis), mom5,
                    log_vol_ratio, basis_dev, streak, bot_net; L2 from L2_GRID
                    on validation. The only place market, model and bot meet,
                    and it is a model fitted on train (principle 1): never a
                    hand-weighted formula. It is compared held-out with each
                    of its components, and the raw probabilities are printed
                    side by side.
    Hyperparameter search, train only: structural's 36 candidates; 4 L2
    values each for features and combiner. Nothing else is tuned.
  - Baselines, not counted as models: `market` (the mid), `asis` (p_asis),
    `base_rate` (the YES rate of usable train rows, one constant).
  - Metrics, held-out only, pooled over the three horizons:
      Brier and log loss (ks.brier, ks.log_loss), mean per row.
      Skill vs a baseline = 1 - loss(model) / loss(baseline), for each metric.
      Its interval is skill +/- z * SE, SE from ks.block_bootstrap (UTC-day
      blocks) of the skill's per-row linearization (see skill()); n_eff is
      that bootstrap's. A contract's three horizons fall in one day block.
      Reliability: per ks.BUCKET_EDGES bucket of the forecast, mean forecast
      vs YES rate with a day-block interval, n and n_eff; a bucket with n_eff
      under MIN_BUCKET_N_EFF is flagged small.
      Sharpness: sd of the forecast and mean |forecast - 0.5|.
      Calibration: ks.calibration_bootstrap's intercept and slope (0 and 1
      when calibrated) with day-block SEs.
  - Multiple comparisons: FAMILY_TESTS = 54 skill intervals share one
    Bonferroni z (ks.bonferroni_z): 7 models x 3 baselines x 2 metrics = 42,
    plus the component comparisons, x 2 metrics: combiner vs structural,
    features, platt_market, bot_only (8) and features vs structural, bot_only
    (4). Reliability intervals use the same z and are descriptive.
  - Verdict rule: a model is called BETTER than a reference only if its
    held-out skill interval lies entirely above zero at that z, WORSE if
    entirely below, otherwise "not distinguishable". Nothing else is a claim.
  - Descriptive, not counted: per-horizon scores; the tuning table.
  - Not tested, so not ruled out: per-second or multi-venue prices, Kalshi's
    own index feed, order-book inputs (no history), tree or neural models,
    horizons other than 10 / 5 / 2 minutes.
"""
import math

import numpy as np
import pandas as pd

import kalshi
import kalshi_strategy_study as st
import kalshi_study as ks

# --- pre-registered design (see module docstring) ---------------------------
VALIDATION_FRACTION = 0.25
VOL_KINDS = ("rv120", "ewma10", "ewma30", "ewma60", "bipower120", "ewma30_seasonal")
EWMA_HALF_LIVES = {"ewma10": 10, "ewma30": 30, "ewma60": 60}
SEASONAL_HALF_LIFE = 30
BIPOWER_WINDOW = 120
BIPOWER_MIN = 60
SPOTS = ("coinbase", "basis_adjusted")
TAIL_DFS = (None, 5, 3)             # None: normal
L2_GRID = (0.1, 1.0, 10.0, 100.0)
PLATT_L2 = 0.0
MOMENTUM_MIN = 5
LOGIT_EPS = 1e-6
PROB_CLIP = 0.001
MIN_BUCKET_N_EFF = 200
MINUTES_PER_YEAR = 365.25 * 24 * 60

MODELS = ("platt_asis", "iso_asis", "structural", "features", "bot_only", "platt_market", "combiner")
BASELINES = ("market", "asis", "base_rate")
METRICS = ("brier", "log_loss")
COMPONENT_COMPARISONS = (("combiner", "structural"), ("combiner", "features"), ("combiner", "platt_market"),
                         ("combiner", "bot_only"), ("features", "structural"), ("features", "bot_only"))
FAMILY_TESTS = (len(MODELS) * len(BASELINES) + len(COMPONENT_COMPARISONS)) * len(METRICS)
EXTRA_FEATURES = ("mom5", "log_vol_ratio", "basis_dev", "streak", "bot_net")


# --- small numerics -------------------------------------------------------------

def logit(p, eps: float = LOGIT_EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


_erf = np.vectorize(math.erf, otypes=[float])


def norm_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def t_cdf(x, df: int):
    """Student-t CDF for df 3 or 5, in closed form (no scipy in this project)."""
    x = np.asarray(x, dtype=float)
    if df == 3:
        u = x / math.sqrt(3.0)
        return 0.5 + (u / (1 + u * u) + np.arctan(u)) / math.pi
    if df == 5:
        u = x / math.sqrt(5.0)
        w = 1 + u * u
        return 0.5 + (u / w * (1 + 2 / (3 * w)) + np.arctan(u)) / math.pi
    raise ValueError("df must be 3 or 5")


def tail_probability(z, df=None):
    """
    P(YES) from the model's z-score. df None: the normal CDF (the model as it
    is). Otherwise a Student-t with `df` degrees of freedom rescaled to unit
    variance, so only the shape of the tails changes, not the spread.
    """
    z = np.asarray(z, dtype=float)
    if df is None:
        return norm_cdf(z)
    return t_cdf(z * math.sqrt(df / (df - 2.0)), df)


def settlement_z(spot, strike, minutes: float, annual_vol):
    """
    The z-score inside kalshi.settlement_probability, elementwise:
    (ln(S/K) - v/2) / sqrt(v), v = sigma^2 * settlement_variance_seconds / year.
    norm_cdf of it IS settlement_probability (pinned by a test).
    """
    spot, strike, vol = (np.asarray(a, dtype=float) for a in (spot, strike, annual_vol))
    seconds = kalshi.settlement_variance_seconds(minutes * 60.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = vol ** 2 * seconds / (MINUTES_PER_YEAR * 60.0)
        return (np.log(spot / strike) - 0.5 * var) / np.sqrt(var)


# --- volatility estimators, each aligned to the minute grid -------------------------

def log_returns(close) -> np.ndarray:
    """r[i] = ln(close[i] / close[i-1]); NaN where either minute is missing."""
    c = np.asarray(close, dtype=float)
    r = np.full(len(c), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = np.log(c[1:] / c[:-1])
    return r


def ewma_variance(r, half_life: float) -> np.ndarray:
    """
    var[i] = sum_k w^k r[i-k]^2 / sum_k w^k over the returns that exist, with
    w = 0.5^(1/half_life): a missing minute contributes nothing but still ages
    the older ones. Uses r[0..i] only. NaN until a return has been seen.
    """
    w = 0.5 ** (1.0 / half_life)
    out = np.full(len(r), np.nan)
    num = den = 0.0
    for i, x in enumerate(np.asarray(r, dtype=float)):
        num *= w
        den *= w
        if x == x:
            num += x * x
            den += 1.0
        if den > 0:
            out[i] = num / den
    return out


def bipower_variance(r, window: int = BIPOWER_WINDOW, min_pairs: int = BIPOWER_MIN) -> np.ndarray:
    """
    Jump-robust per-minute variance: (pi/2) * mean of |r[k]| * |r[k-1]| over
    the pairs ending in (i - window, i]. One large return multiplies two
    ordinary neighbours rather than being squared. NaN under `min_pairs` pairs.
    """
    a = np.abs(np.asarray(r, dtype=float))
    prod = np.full(len(a), np.nan)
    prod[1:] = a[1:] * a[:-1]
    s = pd.Series(prod)
    mean = s.rolling(window, min_periods=min_pairs).mean().to_numpy()
    return (math.pi / 2.0) * mean


def seasonal_factors(ts, r, before_ts: int) -> np.ndarray:
    """
    24 hour-of-day (UTC) volatility factors from returns of candles starting
    before `before_ts`: sqrt(mean r^2 in the hour / mean r^2 overall). 1.0 for
    an hour with no data.
    """
    ts, r = np.asarray(ts, dtype=np.int64), np.asarray(r, dtype=float)
    keep = (ts < before_ts) & ~np.isnan(r)
    hours = (ts[keep] // 3600) % 24
    sq = r[keep] ** 2
    overall = sq.mean() if len(sq) else float("nan")
    out = np.ones(24)
    for h in range(24):
        m = sq[hours == h]
        if len(m) and overall > 0:
            out[h] = math.sqrt(m.mean() / overall)
    return out


def vol_table(grid, seasonal_before_ts: int) -> dict:
    """
    {kind: annualized vol per grid minute} for every estimator except rv120
    (which is ks.model_inputs', taken from ks.model_observations). Entry i uses
    candles up to and including the one starting at grid.t0 + 60 i.
    """
    close = grid.cols["close"]
    r = log_returns(close)
    ts = grid.t0 + 60 * np.arange(len(close), dtype=np.int64)
    ann = math.sqrt(MINUTES_PER_YEAR)
    out = {k: np.sqrt(ewma_variance(r, hl)) * ann for k, hl in EWMA_HALF_LIVES.items()}
    out["bipower120"] = np.sqrt(bipower_variance(r)) * ann
    factors = seasonal_factors(ts, r, seasonal_before_ts)
    f = factors[(ts // 3600) % 24]
    out["ewma30_seasonal"] = np.sqrt(ewma_variance(r / f, SEASONAL_HALF_LIFE)) * f * ann
    return out


# --- the bot's own read ------------------------------------------------------------

def bot_net_votes(candles: pd.DataFrame, quote_ts, cfg=None) -> np.ndarray:
    """
    For each quote time: bullish minus bearish votes of
    signals.evaluate_price_only on the candles that had closed by then (the
    newest starts at quote_ts - 60). NaN when that candle is missing or its
    indicators are not warmed up. `candles` has ts, open, high, low, close,
    volume; indicators are computed once -- every one of them looks backward
    only, so a row's values do not depend on later rows (pinned by a test).
    """
    import config
    import indicators
    import signals
    cfg = cfg or config
    df = indicators.compute_all(candles.sort_values("ts").reset_index(drop=True), cfg)
    pos = {int(t): i for i, t in enumerate(df["ts"].to_numpy())}
    need = [f"sma_{cfg.SMA_SLOW}", f"ema_{cfg.EMA_SLOW}", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"]
    ready = df[need].notna().all(axis=1).to_numpy()
    out = np.full(len(quote_ts), np.nan)
    for k, t in enumerate(np.asarray(quote_ts, dtype=np.int64)):
        i = pos.get(int(t) - 60)
        if i is None or i < 1 or not ready[i]:
            continue
        snap = signals.evaluate_price_only(df.iloc[i - 1:i + 1], cfg)
        out[k] = snap.bullish_count - snap.bearish_count
    return out


# --- the dataset ---------------------------------------------------------------------

def z_column(kind: str, spot: str) -> str:
    return f"z_{kind}_{spot}"


def split_cutoffs(markets: list) -> tuple:
    """(validation cutoff, held-out cutoff) by close time, both from ks.split_cutoff."""
    closes = np.array([m["close_ts"] for m in markets], dtype=np.int64)
    test_cut = ks.split_cutoff(closes)
    val_cut = ks.split_cutoff(closes[closes < test_cut], VALIDATION_FRACTION)
    return val_cut, test_cut


def build_dataset(markets: list, candles_by_ticker: dict, grid, coinbase: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    One row per (contract, horizon) with the outcome, the three baselines'
    inputs, every feature, `split` (fit / validation / test) and `usable`.
    """
    val_cut, test_cut = split_cutoffs(markets)
    d = ks.model_observations(ks.observations(markets, candles_by_ticker), grid)
    d = d.rename(columns={"p_model": "p_asis"})
    d["split"] = np.where(d["close_ts"] >= test_cut, "test", np.where(d["close_ts"] >= val_cut, "validation", "fit"))

    close = grid.cols["close"]
    idx = ((d["quote_ts"].to_numpy(dtype=np.int64) - 60 - grid.t0) // 60)
    ok = (idx >= MOMENTUM_MIN) & (idx < len(close))
    safe = np.where(ok, idx, 0)
    spot, strike = d["spot"].to_numpy(dtype=float), d["strike"].to_numpy(dtype=float)
    proxy = np.array([grid.minute_average_before(t) or np.nan for t in d["open_ts"]], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        plausible = np.abs(strike / spot - 1.0) <= kalshi.MAX_STRIKE_DISTANCE + 1e-9
        spots = {"coinbase": spot, "basis_adjusted": spot * strike / proxy}
        vols = {k: np.where(ok, v[safe], np.nan) for k, v in vol_table(grid, val_cut).items()}
        vols["rv120"] = d["vol"].to_numpy(dtype=float)
        for kind in VOL_KINDS:
            for s in SPOTS:
                z = np.full(len(d), np.nan)
                for h in ks.HORIZONS_MIN:
                    rows = (d["horizon"] == h).to_numpy()
                    z[rows] = settlement_z(spots[s][rows], strike[rows], h, vols[kind][rows])
                d[z_column(kind, s)] = z
        per_minute = vols["rv120"] / math.sqrt(MINUTES_PER_YEAR)
        past = np.where(ok, close[np.maximum(safe - MOMENTUM_MIN, 0)], np.nan)
        d["mom5"] = np.log(np.where(ok, close[safe], np.nan) / past) / (per_minute * math.sqrt(MOMENTUM_MIN))
        d["log_vol_ratio"] = np.log(vols["ewma10"] / vols["rv120"])
    basis, streak = st.basis_features(markets, grid), st.streak_features(markets)
    d["basis_dev"] = d["ticker"].map(basis).fillna(0.0)
    d["streak"] = d["ticker"].map(streak).fillna(0.0)
    d["bot_net"] = bot_net_votes(coinbase, d["quote_ts"].to_numpy(), cfg)

    needed = ["p_asis", "mid", "mom5", "log_vol_ratio", "bot_net"] + [z_column(k, s) for k in VOL_KINDS for s in SPOTS]
    finite = np.isfinite(d[needed].to_numpy(dtype=float)).all(axis=1)
    d["usable"] = d["valid"].to_numpy() & plausible & finite
    return d


# --- models --------------------------------------------------------------------------

def fit_logistic(X, y, l2: float = 0.0, iterations: int = 100) -> dict:
    """
    Logistic regression by Newton's method: minimizes the summed negative log
    likelihood + (l2 / 2) * |w|^2 on inputs standardized by their own mean and
    sd (a constant column gets sd 1); the intercept is not penalized.
    """
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(y)), (X - mean) / sd])
    pen = np.eye(Z.shape[1]) * (l2 + 1e-9)
    pen[0, 0] = 0.0
    beta = np.zeros(Z.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -35, 35)))
        grad = Z.T @ (y - p) - pen @ beta
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + pen + np.eye(Z.shape[1]) * 1e-12
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"beta": beta, "mean": mean, "sd": sd}


def predict_logistic(model: dict, X) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(-1, len(model["mean"]))
    Z = np.column_stack([np.ones(len(X)), (X - model["mean"]) / model["sd"]])
    p = 1 / (1 + np.exp(-np.clip(Z @ model["beta"], -35, 35)))
    return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def fit_isotonic(x, y) -> dict:
    """
    Isotonic (non-decreasing) least-squares regression of y on x by pooling
    adjacent violators; equal x are pooled first. Returns the knots.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=y) / w
    vals, wts, counts = [], [], []
    for v, wt in zip(m, w):
        vals.append(v)
        wts.append(wt)
        counts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            c2 = counts[-2] + counts[-1]
            del vals[-2:], wts[-2:], counts[-2:]
            vals.append(v2)
            wts.append(tw)
            counts.append(c2)
    return {"x": ux, "y": np.repeat(vals, counts)}


def predict_isotonic(model: dict, x) -> np.ndarray:
    """Linear interpolation between the knots, flat beyond them, clipped like every fitted model."""
    return np.clip(np.interp(np.asarray(x, dtype=float), model["x"], model["y"]), PROB_CLIP, 1 - PROB_CLIP)


def structural_raw(d: pd.DataFrame, choice: tuple) -> np.ndarray:
    """The un-recalibrated structural probability for a (vol kind, spot, tail df) choice."""
    kind, spot, df = choice
    return tail_probability(d[z_column(kind, spot)].to_numpy(dtype=float), df)


def design_matrix(d: pd.DataFrame, model: str, structural_choice=None) -> np.ndarray:
    """The inputs of each logistic model, in a fixed column order."""
    if model == "platt_asis":
        return logit(d["p_asis"])[:, None]
    if model == "platt_market":
        return logit(d["mid"])[:, None]
    if model == "bot_only":
        return d[["bot_net"]].to_numpy(dtype=float)
    s = logit(structural_raw(d, structural_choice))
    if model == "structural":
        return s[:, None]
    extra = d[list(EXTRA_FEATURES)].to_numpy(dtype=float)
    if model == "features":
        return np.column_stack([s, extra])
    if model == "combiner":
        return np.column_stack([logit(d["mid"]), s, logit(d["p_asis"]), extra])
    raise ValueError(f"unknown model {model!r}")


def _fit_one(train: pd.DataFrame, model: str, structural_choice=None, l2: float = PLATT_L2) -> dict:
    """One fitted model: a separate fit per horizon."""
    per_h = {}
    for h in ks.HORIZONS_MIN:
        g = train[train["horizon"] == h]
        if model == "iso_asis":
            per_h[h] = fit_isotonic(g["p_asis"], g["outcome"])
        else:
            per_h[h] = fit_logistic(design_matrix(g, model, structural_choice), g["outcome"], l2)
    return {"model": model, "structural_choice": structural_choice, "l2": l2, "per_horizon": per_h}


def predict_one(fitted: dict, d: pd.DataFrame) -> np.ndarray:
    out = np.full(len(d), np.nan)
    for h, m in fitted["per_horizon"].items():
        rows = (d["horizon"] == h).to_numpy()
        if not rows.any():
            continue
        g = d[rows]
        if fitted["model"] == "iso_asis":
            out[rows] = predict_isotonic(m, g["p_asis"])
        else:
            out[rows] = predict_logistic(m, design_matrix(g, fitted["model"], fitted["structural_choice"]))
    return out


def _val_loss(fit: pd.DataFrame, val: pd.DataFrame, model: str, choice, l2) -> float:
    return float(ks.log_loss(predict_one(_fit_one(fit, model, choice, l2), val), val["outcome"]).mean())


def tune_and_fit(dataset: pd.DataFrame) -> dict:
    """
    The whole train-side procedure; reads no row whose split is "test".
    Returns {"models": {name: fitted}, "base_rate": float, "structural_choice",
    "l2": {model: value}, "tuning": DataFrame of every candidate's validation
    log loss}.
    """
    d = dataset[dataset["usable"] & (dataset["split"] != "test")]
    fit, val = d[d["split"] == "fit"], d[d["split"] == "validation"]
    tuning = []
    best = None
    for kind in VOL_KINDS:
        for spot in SPOTS:
            for df in TAIL_DFS:
                loss = _val_loss(fit, val, "structural", (kind, spot, df), PLATT_L2)
                tuning.append({"model": "structural", "setting": f"{kind} / {spot} / tail {df or 'normal'}", "val_log_loss": loss})
                if best is None or loss < best[0] - 1e-12:
                    best = (loss, (kind, spot, df))
    choice = best[1]
    l2 = {}
    for model in ("features", "combiner"):
        best = None
        for lam in L2_GRID:
            loss = _val_loss(fit, val, model, choice, lam)
            tuning.append({"model": model, "setting": f"l2 {lam:g}", "val_log_loss": loss})
            if best is None or loss < best[0] - 1e-12:
                best = (loss, lam)
        l2[model] = best[1]
    models = {m: _fit_one(d, m, choice if m in ("structural", "features", "combiner") else None, l2.get(m, PLATT_L2))
              for m in MODELS}
    return {"models": models, "base_rate": float(d["outcome"].mean()), "structural_choice": choice, "l2": l2,
            "tuning": pd.DataFrame(tuning)}


def predictions(fitted: dict, d: pd.DataFrame) -> pd.DataFrame:
    """Every baseline's and model's P(YES) for the rows of `d`, side by side, never combined."""
    out = d[["ticker", "close_ts", "day", "horizon", "outcome"]].copy()
    out["market"] = d["mid"].to_numpy(dtype=float)
    out["asis"] = d["p_asis"].to_numpy(dtype=float)
    out["base_rate"] = fitted["base_rate"]
    for m in MODELS:
        out[m] = predict_one(fitted["models"][m], d)
    return out


# --- metrics -------------------------------------------------------------------------

LOSSES = {"brier": ks.brier, "log_loss": ks.log_loss}


def skill(loss_model, loss_ref, days, z: float) -> dict:
    """
    Skill = 1 - mean(loss_model) / mean(loss_ref): the share of the
    reference's loss the model removes (positive: the model is better).
    Interval: skill +/- z * SE, with SE the ks.block_bootstrap (day-block) SE
    of the mean of the skill's linearization, per row
        v_i = -(loss_model_i - R * loss_ref_i) / mean(loss_ref),  R = the ratio,
    i.e. the delta method for a ratio of two means, so no second bootstrap is
    needed (checked against a brute-force ratio bootstrap in the tests).
    """
    lm, lr = np.asarray(loss_model, dtype=float), np.asarray(loss_ref, dtype=float)
    ratio = lm.mean() / lr.mean()
    v = -(lm - ratio * lr) / lr.mean()
    b = ks.block_bootstrap(v, days)
    s = 1.0 - ratio
    lo, hi = s - z * b["se"], s + z * b["se"]
    verdict = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "not distinguishable"
    return {"skill": float(s), "se": b["se"], "lo": float(lo), "hi": float(hi), "n": b["n"], "n_eff": b["n_eff"],
            "verdict": verdict}


def scorecard(pred: pd.DataFrame, z: float, calibration_resamples: int = 300) -> pd.DataFrame:
    """Per forecast (baselines first): Brier, log loss, sharpness, calibration, and skill vs each baseline."""
    y, days = pred["outcome"].to_numpy(dtype=float), pred["day"].to_numpy()
    rows = []
    for name in BASELINES + MODELS:
        p = pred[name].to_numpy(dtype=float)
        row = {"forecast": name, "kind": "baseline" if name in BASELINES else "model", "n": len(p),
               "brier": float(ks.brier(p, y).mean()), "log_loss": float(ks.log_loss(p, y).mean()),
               "sharp_sd": float(p.std()), "sharp_abs": float(np.abs(p - 0.5).mean())}
        row.update({"cal_a": float("nan"), "cal_b": float("nan"), "cal_a_se": float("nan"), "cal_b_se": float("nan")})
        if np.ptp(p) > 1e-9:                     # a constant has no slope to fit (its sd can be float residue, not 0)
            row["cal_a"], row["cal_b"] = ks.calibration_fit(p, y)
            try:
                c = ks.calibration_bootstrap(p, y, days, resamples=calibration_resamples)
                row.update({"cal_a_se": c["a_se"], "cal_b_se": c["b_se"]})
            except np.linalg.LinAlgError:        # a resample of very few days can be constant: no SE, not a crash
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def skill_table(pred: pd.DataFrame, z: float) -> pd.DataFrame:
    """Every counted skill interval: FAMILY_TESTS rows."""
    y, days = pred["outcome"].to_numpy(dtype=float), pred["day"].to_numpy()
    pairs = [(m, b, "baseline") for m in MODELS for b in BASELINES] + [(m, c, "component") for m, c in COMPONENT_COMPARISONS]
    rows = []
    for model, ref, kind in pairs:
        for metric in METRICS:
            f = LOSSES[metric]
            s = skill(f(pred[model], y), f(pred[ref], y), days, z)
            rows.append({"model": model, "reference": ref, "kind": kind, "metric": metric, **s})
    return pd.DataFrame(rows)


def reliability_table(p, outcome, days, z: float) -> pd.DataFrame:
    """Per ks.BUCKET_EDGES bucket of the forecast: mean forecast, YES rate +/- z * day-block SE, n, n_eff, small flag."""
    p, y, days = np.asarray(p, dtype=float), np.asarray(outcome, dtype=float), np.asarray(days)
    b = ks.bucket_of(p)
    rows = []
    for i in range(len(ks.BUCKET_EDGES) - 1):
        take = b == i
        if not take.any():
            continue
        s = ks.block_bootstrap(y[take], days[take])
        lo, hi = ks.interval(s, z)
        rows.append({"bucket": f"{ks.BUCKET_EDGES[i]:.1f}-{ks.BUCKET_EDGES[i + 1]:.1f}", "n": int(take.sum()),
                     "n_eff": s["n_eff"], "forecast": float(p[take].mean()), "yes_rate": s["mean"],
                     "lo": max(lo, 0.0) if lo == lo else lo, "hi": min(hi, 1.0) if hi == hi else hi,
                     "small": not (s["n_eff"] >= MIN_BUCKET_N_EFF)})
    return pd.DataFrame(rows)


def by_horizon(pred: pd.DataFrame) -> pd.DataFrame:
    """Brier and log loss of every forecast per horizon. Descriptive: no interval, no claim."""
    rows = []
    for h, g in pred.groupby("horizon"):
        y = g["outcome"].to_numpy(dtype=float)
        for name in BASELINES + MODELS:
            rows.append({"horizon": int(h), "forecast": name, "n": len(g),
                         "brier": float(ks.brier(g[name], y).mean()), "log_loss": float(ks.log_loss(g[name], y).mean())})
    return pd.DataFrame(rows)
