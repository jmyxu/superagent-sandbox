"""打板腿(首板回调v2) TP×SL 参数网格 —— 独立实现, 与基线对照
规则与基线一致: 评分门槛12 / 开盘买 / T+1买日禁卖 / 日级NAV+槽位K2+现金约束 /
涨停骑乘 / 止损收盘判 / 最长8日. 报告每格 胜率/年化/回撤/成交数.

数据: 日线 D:\\tsp-data\\kline_daily_enriched (2021-01~2026-07)
     分钟 C:\\Users\\j6752\\.dsh\\superagent-tools\\min_bars_cache.json (2022-08~2026-07)
"""
import glob, json, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = r"D:\tsp-data\kline_daily_enriched"
MIN_CACHE = r"C:\Users\j6752\.dsh\superagent-tools\min_bars_cache.json"
D0, D1 = "2021-01-01", "2026-07-13"     # 与基线同数据窗
START = pd.Timestamp("2022-08-01")      # 分钟缓存起点 → 回测起点
GATE = 12
MAXD = 8
K = 2
TPS = (0.025, 0.03, 0.035, 0.04)
SLS = (-0.02, -0.03, -0.04)

t0 = time.time()


def load_daily():
    parts = sorted(glob.glob(DATA_DIR + r"\date=*"))
    frames = []
    for p in parts:
        d = p.split("date=")[1]
        if d < D0 or d > D1:
            continue
        f = glob.glob(p + r"\*.parquet")
        if f:
            frames.append(pq.read_table(
                f[0], columns=["symbol", "date", "open", "close", "volume"]).to_pandas())
    raw = pd.concat(frames, ignore_index=True)
    close = raw.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    openp = raw.pivot_table(index="date", columns="symbol", values="open", aggfunc="first")
    vol = raw.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
    for df in (close, openp, vol):
        df.index = pd.to_datetime(df.index)
    return close, openp, vol


def make_mask(close, openp, vol):
    """信号: 昨首板 + 今日回调(-5~3%) + 缩量 + 温和 + 评分>=12 + 连续交易"""
    chg = close.ffill().pct_change(fill_method=None)   # 与基线 pct_change() 默认前填等价
    first_board = (chg.shift(1) >= 0.098) & (chg.shift(2) < 0.098) \
        & (openp.shift(1) < close.shift(1))
    pullback = (chg < 0.098) & (chg >= -0.05) & (chg <= 0.03)
    v5 = vol.rolling(5).mean()
    shrink = vol < v5 * 0.8
    mild = (chg >= -0.03) & (chg <= 0.03)
    lu_mkt = (chg >= 0.098).sum(axis=1)
    lu_ma5 = lu_mkt.rolling(5).mean()
    emo = pd.Series(1.0, index=close.index)
    emo[lu_ma5 >= 30] = 4.0
    emo[(lu_ma5 >= 15) & (lu_ma5 < 30)] = 3.0
    emo[(lu_ma5 >= 8) & (lu_ma5 < 15)] = 2.0
    hot = pd.Series(1.0, index=close.index)
    hot[lu_mkt >= 60] = 3.0
    hot[(lu_mkt >= 30) & (lu_mkt < 60)] = 2.0
    score = shrink.astype(float) * 4 + mild.astype(float) * 4
    score = score.add(emo, axis=0).add(hot, axis=0)
    trading = vol.notna() & (vol > 0)
    cont = trading.rolling(10).min().astype(bool)
    sig = first_board & pullback & (score >= GATE) & cont & trading
    return chg, sig


def build_sigs(sig, openp, dates, start_i):
    """信号日 → 次日开盘买入"""
    out = []
    for i in range(start_i, len(dates) - 1):
        s = sig.iloc[i]
        s = s[s]
        if s.empty:
            continue
        for c_ in s.index:
            e = openp.iloc[i + 1][c_]
            if pd.isna(e) or e <= 0:
                continue
            out.append({"code": str(c_).replace(".SH", "").replace(".SZ", ""),
                        "full": str(c_), "entry": float(e), "start_i": i + 1})
    return out


def precompute_days(sigs, dates, close, openp, chg, get_min):
    """每笔信号预取持仓期(入场次日起最长8日)的日线+分钟数据, 供12组参数复用"""
    for sg in sigs:
        i = sg["start_i"]
        days = []
        for k in range(1, MAXD + 1):
            j = i + k
            if j >= len(dates):
                break
            c_day = close.iloc[j][sg["full"]]
            if pd.isna(c_day):
                continue
            o_day = openp.iloc[j][sg["full"]]
            chg_day = chg.iloc[j][sg["full"]]
            if pd.isna(o_day) or o_day <= 0:
                continue
            bars = get_min(sg["code"], str(dates[j].date()))
            days.append({"j": j, "o": float(o_day), "c": float(c_day),
                         "chg": None if pd.isna(chg_day) else float(chg_day),
                         "bars": bars})
        sg["days"] = days
    return sigs


def simulate_one(sg, tp, sl):
    """单笔独立退出: TP冲高卖/止损(盘中或收盘判)/涨停骑乘/最长8日, 返回(入场日,出场日,收益)"""
    i = sg["start_i"]
    entry = sg["entry"]
    tp_ref = entry * (1 + tp)
    sl_ref = entry * (1 + sl)
    for k, d in enumerate(sg["days"], start=1):
        o, c, chgd, bars = d["o"], d["c"], d["chg"], d["bars"]
        if chgd is not None and chgd >= 0.098:      # 涨停骑乘: 当日涨停则继续持有
            continue
        hit = None
        if bars:
            for p in bars:
                if p >= d["o"] * (1 + tp):          # 盘中冲高 → TP
                    hit = tp
                    break
                if p <= sl_ref:                     # 盘中破位 → SL
                    hit = sl
                    break
        if hit is not None:
            return (i, d["j"], hit)
        if c / entry - 1 <= sl:                     # 收盘判止损
            return (i, d["j"], sl)
        if k == MAXD:                               # 最长8日 → 收盘平仓
            return (i, d["j"], c / entry - 1)
    return None


def portfolio(trades, start_i, n_days, years):
    """日级NAV: 槽位K2 + 现金约束, 每仓1/K资金, 持仓按成本计价(与基线同口径)"""
    cash = 1.0
    pos = {}
    nav = []
    fills = 0
    tq = sorted(trades, key=lambda x: x[0])
    nxt = 0
    for i in range(start_i, n_days):
        for tid in list(pos.keys()):
            _, xj, r = pos[tid]
            if i >= xj:
                cash += (1.0 / K) * (1 + r)
                fills += 1
                del pos[tid]
        while tq and tq[0][0] <= i:
            t = tq.pop(0)
            if t[0] == i and len(pos) < K and cash >= 1.0 / K:
                cash -= 1.0 / K
                pos[id(t)] = t
        nav.append(cash + (1.0 / K) * len(pos))
    nav = np.array(nav)
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    ann = nav[-1] ** (1 / years) - 1
    return ann, dd, fills


def main():
    close, openp, vol = load_daily()
    chg, sig = make_mask(close, openp, vol)
    dates = list(close.index)
    start_i = next(i for i, d in enumerate(dates) if d >= START)
    years = (dates[-1] - START).days / 365.25
    cache = json.load(open(MIN_CACHE))

    def get_min(code, date):
        return cache.get(f"{code}|{date}", [])

    sigs = build_sigs(sig, openp, dates, start_i)
    precompute_days(sigs, dates, close, openp, chg, get_min)
    print(f"评分门槛{GATE}: 信号 {len(sigs)} 笔 | 回测 {START.date()}~{dates[-1].date()} "
          f"({years:.2f}年) | 槽位K{K} 最长{MAXD}日", flush=True)

    results = []
    rows = {}
    for tp in TPS:
        for sl in SLS:
            trades = [t for sg in sigs
                      if (t := simulate_one(sg, tp, sl)) is not None]
            ann, dd, fills = portfolio(trades, start_i, len(dates), years)
            rets = np.array([t[2] for t in trades])
            wr = (rets > 0).mean() * 100
            results.append(dict(tp=tp, sl=sl, wr=wr, ann=ann, dd=dd, fills=fills))
            rows.setdefault(tp, {})[sl] = (wr, ann, dd, fills)
            print(f"TP{tp*100:.1f}% SL{sl*100:.0f}%: 胜率{wr:5.1f}% "
                  f"年化{ann*100:+7.1f}% 回撤{dd*100:6.1f}% 成交{fills:4d} 交易{len(trades):4d}",
                  flush=True)

    print()
    print("结果矩阵 (胜率% / 年化% / 回撤% / 成交数):")
    hdr = "TP\\SL    " + "".join(f"SL{sl*100:+.0f}%        " for sl in SLS)
    print(hdr)
    for tp in TPS:
        line = f"TP{tp*100:.1f}%  "
        for sl in SLS:
            wr, ann, dd, fills = rows[tp][sl]
            line += f"{wr:4.0f}/{ann*100:+5.1f}/{dd*100:4.1f}/{fills:4d}   "
        print(line)

    best_ann = max(results, key=lambda r: r["ann"])
    best_wr = max(results, key=lambda r: r["wr"])
    print(f"\n年化最优: TP{best_ann['tp']*100:.1f}% SL{best_ann['sl']*100:.0f}% → "
          f"胜率{best_ann['wr']:.1f}% 年化{best_ann['ann']*100:+.1f}% "
          f"回撤{best_ann['dd']*100:.1f}% 成交{best_ann['fills']}")
    print(f"胜率最优: TP{best_wr['tp']*100:.1f}% SL{best_wr['sl']*100:.0f}% → "
          f"胜率{best_wr['wr']:.1f}% 年化{best_wr['ann']*100:+.1f}% "
          f"回撤{best_wr['dd']*100:.1f}% 成交{best_wr['fills']}")
    print(f"用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
