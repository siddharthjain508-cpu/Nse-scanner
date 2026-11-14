"""
CHoCH + EMA-pullback swing strategy engine.

Rule 1  - Weekly Change of Character (CHoCH), ported bar-by-bar from
          "Choch Pattern Levels [BigBeluga]" (Pine v6).
          Original indicator (c) BigBeluga, licensed CC BY-NC-SA 4.0
          https://creativecommons.org/licenses/by-nc-sa/4.0/
          This port is for personal, non-commercial use under the same licence.
Rule 2  - Do NOT enter on the CHoCH. Wait for a pullback that takes support
          on the 20 or 50 EMA, then enter on the close of that support candle.

Pure pandas/numpy - no network, no UI - so it can be tested on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # CHoCH indicator inputs (same defaults as the Pine script)
    pivot_len: int = 10          # "Length"
    max_zones: int = 10          # "Amount of Patterns" (display only)

    # Pullback / entry
    pullback_tf: str = "W"       # "W" = weekly EMAs, "D" = daily EMAs
    ema_fast: int = 20
    ema_slow: int = 50
    touch_tol_pct: float = 1.0   # low may stop this % above the EMA and still count as a touch
    require_green: bool = False  # support candle must close above its open
    min_bars_after_choch: int = 1
    max_wait_bars: int = 26      # give up waiting for a pullback after N bars (0 = never)

    # Risk / exit
    stop_mode: str = "candle_low"    # "candle_low" (support-candle low) or "choch_level" (CHoCH swing low - often very wide on weekly)
    target_r: float = 2.0            # take-profit in R multiples (0 = no target)
    exit_on_bearish_choch: bool = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> weekly bars labelled by the week's Friday (NSE week)."""
    w = daily.resample("W-FRI", label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return w.dropna(subset=["Open", "High", "Low", "Close"])


def ema(series: pd.Series, n: int) -> pd.Series:
    """EMA seeded with an SMA of the first n values, like Pine's ta.ema."""
    v = series.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1] = v[:n].mean()
        a = 2.0 / (n + 1)
        for i in range(n, len(v)):
            out[i] = a * v[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=series.index)


def _is_pivot_high(h: np.ndarray, c: int, n: int) -> bool:
    x = h[c]
    return bool(np.all(x > h[c - n:c]) and np.all(x >= h[c + 1:c + n + 1]))


def _is_pivot_low(l: np.ndarray, c: int, n: int) -> bool:
    x = l[c]
    return bool(np.all(x < l[c - n:c]) and np.all(x <= l[c + 1:c + n + 1]))


# --------------------------------------------------------------------------- #
# Rule 1 - CHoCH detection (faithful port of the Pine logic)
# --------------------------------------------------------------------------- #
def detect_choch(w: pd.DataFrame, pivot_len: int = 10, max_zones: int = 10) -> list[dict]:
    """
    Returns a list of CHoCH events, oldest first. Each event:
      direction  'up' | 'down'
      bar/date   bar where the CHoCH printed
      pivot_*    the swing high (up) / swing low (down) that was broken
      level      lowest low (up) / highest high (down) between pivot and CHoCH bar
      level_date bar of that extreme - the horizontal line on the chart
      delta      delta volume over the pattern
      removed_* when/why the level disappeared from the chart (None = still drawn)
    """
    n = pivot_len
    O, H, L, C, V = (w[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    idx = w.index
    N = len(w)

    ph = pl = np.nan
    ph_i = pl_i = -1
    prev_ph = prev_pl = np.nan
    trend = False                      # Pine: var trend = false
    events: list[dict] = []
    active: list[dict] = []            # zones currently drawn
    body_h = np.maximum(O, C)
    body_l = np.minimum(O, C)

    for t in range(N):
        c = t - n
        if c >= n:
            if _is_pivot_high(H, c, n):
                ph_i, ph = c, H[c]
            if _is_pivot_low(L, c, n):
                pl_i, pl = c, L[c]

        # ---- bullish CHoCH: ta.crossover(high, ph) and not trend
        if (t > 0 and not trend and not np.isnan(ph) and not np.isnan(prev_ph)
                and H[t] > ph and H[t - 1] <= prev_ph):
            seg = slice(ph_i, t + 1)
            lvl = L[seg].min()
            # Pine scans from the current bar backwards -> most recent occurrence
            lvl_bar = ph_i + int(np.where(L[seg] == lvl)[0].max())
            delta = float(np.where(C[seg] > O[seg], V[seg], -V[seg]).sum())
            ev = dict(direction="up", bar=t, date=idx[t], pivot_bar=ph_i, pivot_date=idx[ph_i],
                      pivot_price=float(ph), break_price=float(H[t]), level=float(lvl),
                      level_bar=lvl_bar, level_date=idx[lvl_bar], delta=delta,
                      removed_bar=None, removed_date=None, removed_reason=None)
            events.append(ev)
            active.append(ev)
            trend = True

        # ---- bearish CHoCH: ta.crossunder(low, pl) and trend
        if (t > 0 and trend and not np.isnan(pl) and not np.isnan(prev_pl)
                and L[t] < pl and L[t - 1] >= prev_pl):
            seg = slice(pl_i, t + 1)
            lvl = H[seg].max()
            lvl_bar = pl_i + int(np.where(H[seg] == lvl)[0].max())
            delta = float(np.where(C[seg] > O[seg], V[seg], -V[seg]).sum())
            ev = dict(direction="down", bar=t, date=idx[t], pivot_bar=pl_i, pivot_date=idx[pl_i],
                      pivot_price=float(pl), break_price=float(L[t]), level=float(lvl),
                      level_bar=lvl_bar, level_date=idx[lvl_bar], delta=delta,
                      removed_bar=None, removed_date=None, removed_reason=None)
            events.append(ev)
            active.append(ev)
            trend = False

        # ---- level is removed once a candle BODY trades through it (this bar or previous)
        for z in list(active):
            y = z["level"]
            hit = body_h[t] > y and body_l[t] < y
            if t > 0:
                hit = hit or (body_h[t - 1] > y and body_l[t - 1] < y)
            if hit:
                z.update(removed_bar=t, removed_date=idx[t], removed_reason="mitigated")
                active.remove(z)

        if len(active) > max_zones:
            z = active.pop(0)
            z.update(removed_bar=t, removed_date=idx[t], removed_reason="limit")

        prev_ph, prev_pl = ph, pl

    return events


# --------------------------------------------------------------------------- #
# Rule 2 - pullback to 20/50 EMA after a bullish CHoCH, then manage the trade
# --------------------------------------------------------------------------- #
def simulate_setups(events: list[dict], p: pd.DataFrame, prm: Params) -> list[dict]:
    """
    p = bars on the pullback timeframe with columns Open/High/Low/Close/ema_fast/ema_slow.
    One setup per bullish CHoCH. Life-cycle:
      AWAITING_PULLBACK -> IN_TRADE -> CLOSED
                        -> CANCELLED (bearish CHoCH first) | INVALIDATED (close < CHoCH level) | EXPIRED
    """
    dates = p.index
    O, H, L, C = (p[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    EF, ES = p["ema_fast"].to_numpy(dtype=float), p["ema_slow"].to_numpy(dtype=float)
    tol = prm.touch_tol_pct / 100.0
    bear_dates = [e["date"] for e in events if e["direction"] == "down"]

    setups = []
    for e in events:
        if e["direction"] != "up":
            continue
        nxt_bear = next((d for d in bear_dates if d > e["date"]), None)
        s = dict(choch_date=e["date"], choch_level=e["level"], pivot_price=e["pivot_price"],
                 delta=e["delta"], status="AWAITING_PULLBACK", note="",
                 signal_date=None, ema_used=None, entry=np.nan, stop=np.nan, target=np.nan,
                 exit_date=None, exit_price=np.nan, exit_reason=None, r_multiple=np.nan,
                 signal_bar_index=None)
        j = int(dates.searchsorted(e["date"], side="right"))
        waited = 0
        while j < len(dates):
            d = dates[j]
            if s["status"] == "AWAITING_PULLBACK":
                if nxt_bear is not None and d >= nxt_bear:
                    s.update(status="CANCELLED", note="Bearish CHoCH before any pullback entry")
                    break
                if C[j] < e["level"]:
                    s.update(status="INVALIDATED", note="Closed below the CHoCH level")
                    break
                waited += 1
                if prm.max_wait_bars and waited > prm.max_wait_bars:
                    s.update(status="EXPIRED", note=f"No EMA pullback within {prm.max_wait_bars} bars")
                    break
                if waited >= prm.min_bars_after_choch and j > 0:
                    hit = None
                    # check the slow EMA first: a candle that dips to the 50 is a 50-EMA bounce
                    for name, arr in (("EMA%d" % prm.ema_slow, ES), ("EMA%d" % prm.ema_fast, EF)):
                        em, em_prev = arr[j], arr[j - 1]
                        if np.isnan(em) or np.isnan(em_prev):
                            continue
                        came_from_above = C[j - 1] > em_prev
                        touched = L[j] <= em * (1 + tol)
                        held = C[j] >= em
                        green_ok = (C[j] > O[j]) or not prm.require_green
                        if came_from_above and touched and held and green_ok:
                            hit = name
                            break
                    if hit:
                        entry = C[j]
                        stop = e["level"] if prm.stop_mode == "choch_level" else L[j]
                        if entry > stop:
                            risk = entry - stop
                            s.update(status="IN_TRADE", signal_date=d, ema_used=hit, entry=entry,
                                     stop=stop, signal_bar_index=j,
                                     target=entry + prm.target_r * risk if prm.target_r > 0 else np.nan)
            elif s["status"] == "IN_TRADE":
                stop, tgt = s["stop"], s["target"]
                if L[j] <= stop:
                    s.update(status="CLOSED", exit_date=d, exit_price=min(O[j], stop), exit_reason="STOP")
                    break
                if not np.isnan(tgt) and H[j] >= tgt:
                    s.update(status="CLOSED", exit_date=d, exit_price=max(O[j], tgt), exit_reason="TARGET")
                    break
                if prm.exit_on_bearish_choch and nxt_bear is not None and d >= nxt_bear:
                    s.update(status="CLOSED", exit_date=d, exit_price=C[j], exit_reason="BEARISH CHOCH")
                    break
            j += 1

        if s["status"] in ("IN_TRADE", "CLOSED"):
            risk = s["entry"] - s["stop"]
            px = s["exit_price"] if s["status"] == "CLOSED" else C[-1]
            s["r_multiple"] = (px - s["entry"]) / risk if risk > 0 else np.nan
        setups.append(s)
    return setups


# --------------------------------------------------------------------------- #
# One call per stock
# --------------------------------------------------------------------------- #
def analyze(daily: pd.DataFrame, prm: Params, last_bar_forming: bool = False) -> dict:
    """
    daily: OHLCV indexed by date (daily bars).
    last_bar_forming: True while the latest bar on the pullback timeframe is still open
                      (signals on it are provisional until the candle closes).
    """
    w = daily if prm.pullback_tf == "WEEKLY_INPUT" else to_weekly(daily)
    events = detect_choch(w, prm.pivot_len, prm.max_zones)

    p = (w if prm.pullback_tf in ("W", "WEEKLY_INPUT") else daily[["Open", "High", "Low", "Close", "Volume"]]).copy()
    p["ema_fast"] = ema(p["Close"], prm.ema_fast)
    p["ema_slow"] = ema(p["Close"], prm.ema_slow)
    setups = simulate_setups(events, p, prm)

    last = p.iloc[-1]
    st = dict(status="NO CHOCH", detail="", close=float(last["Close"]),
              ema_fast=float(last["ema_fast"]), ema_slow=float(last["ema_slow"]),
              dist_fast_pct=(last["Close"] / last["ema_fast"] - 1) * 100,
              dist_slow_pct=(last["Close"] / last["ema_slow"] - 1) * 100,
              choch_date=None, choch_level=np.nan, signal_date=None, ema_used=None,
              entry=np.nan, stop=np.nan, target=np.nan, r_now=np.nan, provisional=False,
              new_choch=False, fresh_signal=False, fresh_exit=False)

    if events:
        last_ev = events[-1]
        st["new_choch"] = last_ev["direction"] == "up" and last_ev["bar"] == len(w) - 1
        if last_ev["direction"] == "down":
            st.update(status="BEARISH", detail="Last CHoCH was bearish - no long setup",
                      choch_date=last_ev["date"], choch_level=last_ev["level"])
        else:
            s = setups[-1]
            st.update(choch_date=s["choch_date"], choch_level=s["choch_level"],
                      signal_date=s["signal_date"], ema_used=s["ema_used"], entry=s["entry"],
                      stop=s["stop"], target=s["target"], r_now=s["r_multiple"])
            if s["status"] == "AWAITING_PULLBACK":
                st.update(status="AWAITING PULLBACK",
                          detail="New weekly CHoCH - wait for EMA support" if st["new_choch"]
                          else "Bullish CHoCH - wait for EMA support")
            elif s["status"] == "IN_TRADE":
                fresh = s["signal_bar_index"] == len(p) - 1
                st["fresh_signal"] = fresh
                st["provisional"] = fresh and last_bar_forming
                st.update(status="ENTRY SIGNAL" if fresh else "IN TRADE",
                          detail=f"Support on {s['ema_used']}"
                          + (" (candle still forming - confirm at close)" if st["provisional"] else ""))
            elif s["status"] == "CLOSED":
                st["fresh_exit"] = s["exit_date"] == p.index[-1]
                st.update(status=f"EXITED ({s['exit_reason']})", detail=f"Exit {s['exit_date']:%d-%b-%Y}")
            else:
                st.update(status=s["status"], detail=s["note"])

    return dict(weekly=w, pull=p, events=events, setups=setups, state=st)


def stats(setups: list[dict]) -> dict:
    closed = [s for s in setups if s["status"] == "CLOSED" and not np.isnan(s["r_multiple"])]
    if not closed:
        return dict(trades=0, win_rate=np.nan, avg_r=np.nan, total_r=0.0)
    r = np.array([s["r_multiple"] for s in closed])
    return dict(trades=len(r), win_rate=float((r > 0).mean() * 100), avg_r=float(r.mean()),
                total_r=float(r.sum()))
