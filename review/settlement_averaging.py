"""Run from the repo root: python review/settlement_averaging.py

Model P(YES) (point-in-time N(d2)) vs a second-by-second Monte Carlo of KXBTC15M's real rule:
YES iff the simple average of the 60 one-second prints before close >= strike. Driftless GBM.
For T < 1 minute the already-elapsed part of the window is taken at the current price, which
is an assumption (the real elapsed average is known on the day, not modelled here).
"""
import math, numpy as np, sys
sys.path.insert(0, ".")
import kalshi
MPY = 365.25*24*60
def mc(S, K, T_min, vol, n=200_000, seed=1):
    """P(simple average of the 60 one-second prices before close >= K), driftless GBM."""
    rng = np.random.default_rng(seed)
    dt = 1/60/MPY                         # one second, in years
    T_s = int(round(T_min*60))
    pre = max(T_s-60, 0)                  # seconds before the averaging window opens
    logS = np.full(n, math.log(S))
    if pre:
        tau = pre*dt
        logS += -0.5*vol**2*tau + vol*math.sqrt(tau)*rng.standard_normal(n)
    k = T_s - pre                          # averaging-window seconds still ahead (<=60)
    acc = np.zeros(n)
    for _ in range(k):
        logS += -0.5*vol**2*dt + vol*math.sqrt(dt)*rng.standard_normal(n)
        acc += np.exp(logS)
    # seconds of the window already elapsed are taken at the current price S (best available proxy)
    avg = (acc + (60-k)*S)/60
    p = (avg >= K).mean()
    return p, math.sqrt(p*(1-p)/n)
vol = 0.45
for T in (14, 5, 2, 1, 0.5, 0.25):
    for off_bps in (5, 15):
        S = 75_000.0; K = S/(1+off_bps/1e4)
        pm = kalshi.crossing_probability(S, K, T, vol)
        p, se = mc(S, K, T, vol)
        print(f"T={T:>5} min  spot {off_bps:>2}bp above strike  model={pm:.3f}  averaged(MC)={p:.3f} ±{se:.3f}  gap={p-pm:+.3f}")
