"""
Full-market NSE scan: weekly CHoCH -> pullback to 20/50 EMA.
Runs on GitHub Actions every weekday evening, writes docs/results.json + docs/index.html
(served by GitHub Pages) and optionally sends a Telegram summary.

Settings (environment variables, all optional):
  MIN_PRICE         skip stocks below this price            (default 20)
  MIN_TURNOVER_CR   skip stocks with avg daily traded value (20 days) below this, in Rs crore (default 1)
  HISTORY           price history to download               (default 5y)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID   for phone alerts
If a file watchlist.txt with symbols exists and is not empty, only those symbols are scanned.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from strategy import Params, analyze, stats, to_weekly

IST = timezone(timedelta(hours=5, minutes=30))
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "*/*"}
COLS = ["Open", "High", "Low", "Close", "Volume"]

MIN_PRICE = float(os.getenv("MIN_PRICE", "20"))
MIN_TURNOVER_CR = float(os.getenv("MIN_TURNOVER_CR", "1"))
HISTORY = os.getenv("HISTORY", "5y")
BATCH = 100


# ----------------------------------------------------------------- universe
def nse_symbols() -> list[str]:
    wl = os.path.join(HERE, "watchlist.txt")
    if os.path.exists(wl):
        syms = [s.strip().upper() for s in open(wl) if s.strip() and not s.startswith("#")]
        if syms:
            print(f"Using watchlist.txt ({len(syms)} symbols)")
            return syms
    for url in ("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
                "https://archives.nseindia.com/content/equities/EQUITY_L.csv"):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
            syms = sorted(df["SYMBOL"].astype(str).str.strip().unique())
            print(f"NSE equity list: {len(syms)} symbols")
            return syms
        except Exception as e:
            print("Equity list failed:", url, e)
    url = "https://niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    syms = sorted(pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str).str.strip())
    print(f"Fallback Nifty Total Market list: {len(syms)} symbols")
    return syms


# ----------------------------------------------------------------- prices
def download(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for i in range(0, len(symbols), BATCH):
        chunk = [f"{s}.NS" for s in symbols[i:i + BATCH]]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=HISTORY, interval="1d", auto_adjust=False,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print("download retry", attempt + 1, e)
                time.sleep(15)
        if raw is None or raw.empty:
            continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[COLS].dropna(subset=["Open", "High", "Low", "Close"])
            except KeyError:
                continue
            if len(df) < 250:
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            data[t[:-3]] = df
        print(f"downloaded {min(i + BATCH, len(symbols))}/{len(symbols)}  (usable {len(data)})")
        time.sleep(2)
    return data


def download_all(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = download(symbols)
    missing = [s for s in symbols if s not in data]
    if missing and len(missing) < len(symbols):
        print(f"retrying {len(missing)} symbols that returned nothing")
        time.sleep(30)
        data.update(download(missing))
    return data


# ----------------------------------------------------------------- helpers
def f(x, n=2):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, n)


def d(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def week_forming(label: pd.Timestamp) -> bool:
    close_dt = datetime(label.year, label.month, label.day, 15, 30, tzinfo=IST)
    return datetime.now(IST) < close_dt


def telegram(text: str) -> None:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    for i in range(0, len(text), 3800):
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text[i:i + 3800],
                                "disable_web_page_preview": "true"}, timeout=20)
        except requests.RequestException as e:
            print("telegram failed", e)


# ----------------------------------------------------------------- main
def main():
    prm = Params()  # weekly CHoCH, weekly 20/50 EMA pullback, stop = support-candle low, 2R target
    symbols = nse_symbols()
    data = download_all(symbols)

    rows, skipped_liq = [], 0
    for sym, daily in data.items():
        last20 = daily.tail(20)
        turnover_cr = float((last20["Close"] * last20["Volume"]).mean() / 1e7)
        price = float(daily["Close"].iloc[-1])
        if price < MIN_PRICE or turnover_cr < MIN_TURNOVER_CR:
            skipped_liq += 1
            continue
        try:
            w = to_weekly(daily)
            res = analyze(daily, prm, last_bar_forming=week_forming(w.index[-1]))
        except Exception as e:
            print("analyze failed", sym, e)
            continue
        s, st = res["state"], stats(res["setups"])
        last = res["setups"][-1] if res["setups"] else None
        rows.append(dict(
            symbol=sym, week=d(w.index[-1]), status=s["status"], detail=s["detail"],
            close=f(s["close"]), ema20=f(s["ema_fast"]), ema50=f(s["ema_slow"]),
            vs_ema20=f(s["dist_fast_pct"], 1), vs_ema50=f(s["dist_slow_pct"], 1),
            choch_date=d(s["choch_date"]), choch_level=f(s["choch_level"]),
            signal_date=d(s["signal_date"]), ema_used=s["ema_used"], entry=f(s["entry"]),
            stop=f(s["stop"]), target=f(s["target"]), r_now=f(s["r_now"]),
            provisional=bool(s["provisional"]), new_choch=bool(s["new_choch"]),
            fresh_signal=bool(s["fresh_signal"]), fresh_exit=bool(s["fresh_exit"]),
            exit_reason=last["exit_reason"] if last else None, turnover_cr=f(turnover_cr, 1),
            hist_trades=st["trades"], hist_winrate=f(st["win_rate"], 0), hist_avg_r=f(st["avg_r"])))

    now = datetime.now(IST)
    out = {"scanned_at": now.strftime("%d %b %Y, %I:%M %p IST"),
           "data_week": max((r["week"] for r in rows), default=None),
           "universe": len(symbols), "downloaded": len(data), "skipped_illiquid": skipped_liq,
           "filters": {"min_price": MIN_PRICE, "min_turnover_cr": MIN_TURNOVER_CR},
           "results": rows}
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    shutil.copy(os.path.join(HERE, "dashboard.html"), os.path.join(DOCS, "index.html"))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    print(f"Scanned {len(rows)} stocks ({skipped_liq} skipped as illiquid).")

    # ---- Telegram summary
    buys = [r for r in rows if r["fresh_signal"]]
    chochs = [r for r in rows if r["new_choch"]]
    exits = [r for r in rows if r["fresh_exit"]]
    near = sorted([r for r in rows if r["status"] == "AWAITING PULLBACK"
                   and min(abs(r["vs_ema20"] or 99), abs(r["vs_ema50"] or 99)) <= 3],
                  key=lambda r: min(abs(r["vs_ema20"] or 99), abs(r["vs_ema50"] or 99)))
    tag = "" if now.weekday() == 4 or now.weekday() >= 5 else " (provisional – confirm Friday close)"
    lines = [f"📊 CHoCH scan {now:%d %b} – {len(rows)} stocks"]
    if buys:
        lines.append(f"\n🟢 BUY{tag}:")
        lines += [f"{r['symbol']} {r['ema_used']} entry {r['entry']} SL {r['stop']} T {r['target']}" for r in buys]
    if chochs:
        lines.append(f"\n🔷 New weekly CHoCH{tag}: " + ", ".join(r["symbol"] for r in chochs))
    if exits:
        lines.append("\n🔴 Exits: " + ", ".join(f"{r['symbol']} ({r['exit_reason']})" for r in exits))
    if near:
        lines.append("\n👀 Near EMA (awaiting pullback): " + ", ".join(
            f"{r['symbol']} {min(r['vs_ema20'], r['vs_ema50'], key=abs):+.1f}%" for r in near[:40]))
    if not (buys or chochs or exits):
        lines.append("No new signals today.")
    url = os.getenv("DASHBOARD_URL")
    if url:
        lines.append(f"\n{url}")
    msg = "\n".join(lines)
    print(msg)
    telegram(msg)


if __name__ == "__main__":
    main()
