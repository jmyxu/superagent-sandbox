"""大单资金流因子研究（issue #34）：资金流因子构建 + 日度 IC + 五分位组合回测。

数据口径：
- 资金流：东财免费接口（scripts/fetch_fund_flow.py，落盘 data/fundflow/<代码>.csv）。
  免费口径只提供**最近约 120 个交易日**的个股资金流历史，因此本研究窗口为
  2026-05-21 ~ 2026-11-13（约 6 个月）——这是东财免费数据的硬性上限，结论
  外推需谨慎，见 docs/fund-flow-factor-research.md；
- 行情：data/daily/<代码>_qfq.csv（第 6 步 screener 拉取口径，qfq 前复权，
  东财行情限流时段经 akshare 腾讯源）；腾讯源日线不含成交额，成交额按东财
  净占比定义反推（主力净额 ÷ 主力净占比，两口径并存时核对偏差）。

因子定义（t 日收盘后可用，只用 ≤t 的信息，无未来函数）：
- 资金流强度因子：N 日累计净流入 ÷ N 日累计成交额（tier ∈ 主力 main /
  大单 big / 超大单 super，N ∈ 1/5/10/20），例
  main_5 = Σ(t-4..t) 主力净流入 ÷ Σ(t-4..t) 成交额；
- 同窗口基准因子（对照）：rev5 / rev20（短期反转）、mom20（动量）、
  vol20（低波动，取负号做多低波动）。

评估（全部用同窗口同股票池，保证可比）：
1. 日度横截面 Spearman rank IC：因子(t) vs 未来 h=1/5/10/20 日收益，
   报告天数、均值、t 值、年化 ICIR（独立样本假设，重叠样本会使 t 高估）、
   正占比；
2. 五分位组合（每 5 个交易日调仓）：Top（最高分位）多头、Bottom（最低
   分位）多头、Top-Bottom 多空（假设口径，A 股不能做空）。t 日收盘按因子
   排序，t+1 日起按持仓名单的日收益等权计净值（未建模开盘跳空冲击）；
   成本按每次调仓的买卖权重差计费：买入佣金万三、卖出佣金万三+印花税万五；
   与沪深300、全池等权对照。

用法：
    python scripts/fund_flow_factor.py selftest        # 合成数据自检（不访问网络）
    python scripts/fund_flow_factor.py run             # 全流程（只读本地缓存）
    python scripts/fund_flow_factor.py run --bench     # 同左，联网拉沪深300基准

输出（output/fund_flow_factor/）：
  - factor_stats.csv：因子覆盖与分布统计（天数、均值、标准差、一阶自相关）；
  - factor_corr.csv：因子间横截面平均 Spearman 相关；
  - ic_daily.csv：逐日 IC 明细；ic_summary.csv：IC 汇总；
  - quintile_summary.csv：五分位组合指标汇总；
  - quintile_equity.csv：逐日净值（Top/Bottom/多空 × 因子、基准、全池等权）；
  - ic_compare.png / quintile_compare.png：对比图。

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/fund_flow_factor.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import DISCLAIMER, _setup_cjk_font, load_benchmark  # noqa: E402
from screener import fetch_universe  # noqa: E402

FF_DIR_DEFAULT = "data/fundflow"
DAILY_DIR_DEFAULT = "data/daily"
OUT_DIR_DEFAULT = "output/fund_flow_factor"
ADJUST = "qfq"

TIERS = ("main", "big", "super")
WINDOWS = (1, 5, 10, 20)
HORIZONS = (1, 5, 10, 20)
MIN_N_IC = 80          # IC 单日最少有效股票数
MIN_N_REBAL = 100      # 调仓日最少可用因子数
REBALANCE_EVERY = 5    # 每 5 个交易日调仓
COMMISSION = 0.0003    # 买入佣金万三
STAMP = 0.0005         # 卖出印花税万五

# 资金流因子（tier × 窗口）与基准因子；small 为散户档（小单）对照腿
FLOW_FACTORS = [f"{tier}_{n}" for tier in TIERS for n in WINDOWS] + \
               ["small_5", "small_20"]
BASELINE_FACTORS = ["rev5", "rev20", "mom20", "vol20"]
ALL_FACTORS = FLOW_FACTORS + BASELINE_FACTORS


# ---------------------------------------------------------------------------
# 面板构建
# ---------------------------------------------------------------------------

def _load_daily(symbol: str, datadir: Path) -> pd.DataFrame | None:
    """读日线 CSV（中文列），返回 date/close/amount（amount 可能整列为 NaN）。"""
    csv_path = datadir / f"{symbol}_{ADJUST}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if not {"日期", "收盘"} <= set(df.columns):
        return None
    df = df.rename(columns={"日期": "date", "收盘": "close"}).copy()
    if "成交额" not in df.columns:
        df["amount"] = np.nan
    else:
        df = df.rename(columns={"成交额": "amount"})
    df = df[["date", "close", "amount"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("close", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    return df.reset_index(drop=True)


def build_stock_frame(symbol: str, ff_dir: Path, daily_dir: Path) -> pd.DataFrame | None:
    """合并资金流与日线为单票 DataFrame，并计算收益、资金流因子与基准因子。

    成交额口径：优先用日线 CSV 的 成交额；腾讯源日线不含成交额（整列 NaN），
    此时按东财净占比定义反推 `amount = 主力净额 / (主力净占比/100)`
    （东财资金流口径自洽，误差来自净占比的两位小数舍入）；两者皆无的极少数
    日以该票自身中位成交额填充（占位，仅影响滚窗求和完整性）。反推口径与
    日线口径的偏差用于数据质量核对（factor_stats 附注）。

    返回列：date, close, amount, *_net, *_ratio, ret, fwd_<h>（h=1/5/10/20）、
    全部因子列。数据不足（<20 行）返回 None。
    """
    ff_csv = ff_dir / f"{symbol}.csv"
    if not ff_csv.exists():
        return None
    ff = pd.read_csv(ff_csv, parse_dates=["date"])
    if ff.empty:
        return None
    daily = _load_daily(symbol, daily_dir)
    if daily is None or daily.empty:
        return None
    # 以日线 CSV 为左表：收益用 qfq 日线口径；资金流 CSV 的 close 为不复权
    # 收盘价，仅作参考（列名加 _ff 后缀保留）。
    df = daily.merge(ff, on="date", how="inner", suffixes=("", "_ff"))
    if df.empty:
        return None
    df = df[df["close"] > 0].dropna(subset=["close"]).sort_values("date")
    if len(df) < 20:
        return None

    # 反推成交额（东财净占比定义），并选优：日线成交额 > 反推 > 中位填充
    with np.errstate(invalid="ignore", divide="ignore"):
        implied = df["main_net"] / (df["main_ratio"] / 100.0)
        implied = implied.where(df["main_ratio"].abs() > 0.005)
    amount = df["amount"].where(df["amount"].notna(), implied)
    med_amount = amount.median()
    amount = amount.fillna(med_amount)
    # 数据质量核对：两口径并存的日子，偏差超过 35% 视为口径异常，剔除该票
    both = df["amount"].notna() & implied.notna()
    if both.sum() >= 5:
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = (np.log(df["amount"][both] / implied[both]).abs()).median()
        if np.isfinite(dev) and dev > np.log(1.35):
            return None
    df["amount"] = amount

    df = df.reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    for h in HORIZONS:
        df[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1.0

    for tier in TIERS:
        net = df[f"{tier}_net"]
        for n in WINDOWS:
            df[f"{tier}_{n}"] = net.rolling(n).sum() / df["amount"].rolling(n).sum()
    # 小单（散户）净流入占比：大单因子的对照腿（Hvidkjaer 2008 反转假说）
    small = df["small_net"]
    for n in (5, 20):
        df[f"small_{n}"] = small.rolling(n).sum() / df["amount"].rolling(n).sum()
    df["rev5"] = -(df["close"] / df["close"].shift(5) - 1.0)
    df["rev20"] = -(df["close"] / df["close"].shift(20) - 1.0)
    df["mom20"] = df["close"] / df["close"].shift(20) - 1.0
    df["vol20"] = -df["ret"].rolling(20).std(ddof=1)
    return df


def build_panel(ff_dir: Path, daily_dir: Path, offline: bool = True) -> pd.DataFrame:
    """全池合并为长面板（symbol × date），记录剔除原因统计。"""
    universe = fetch_universe(offline=offline)
    symbols = universe["symbol"].tolist()
    print(f"[ok] 股票池: {len(symbols)} 只（快照 {universe['as_of'].iloc[0]}）")

    frames: dict[str, pd.DataFrame] = {}
    skipped_no_ff = skipped_no_daily = skipped_bad = skipped_short = 0
    for symbol in symbols:
        ff_csv = ff_dir / f"{symbol}.csv"
        if not ff_csv.exists():
            skipped_no_ff += 1
            continue
        daily = _load_daily(symbol, daily_dir)
        if daily is None:
            skipped_no_daily += 1
            continue
        df = build_stock_frame(symbol, ff_dir, daily_dir)
        if df is None:
            merged = pd.read_csv(ff_csv, parse_dates=["date"]).merge(
                daily, on="date", how="inner")
            if merged.empty:
                skipped_no_daily += 1
            elif len(merged) < 20:
                skipped_short += 1
            else:
                skipped_bad += 1  # 成交额两口径偏差超限
            continue
        frames[symbol] = df

    panel = (pd.concat(frames.values(), keys=frames.keys(), names=["symbol"])
             .reset_index(level=0).reset_index(drop=True))
    print(f"[ok] 面板: {len(frames)} 只 × {panel['date'].nunique()} 个交易日；"
          f"剔除——无资金流 {skipped_no_ff}、无日线/无重叠 {skipped_no_daily}、"
          f"成交额口径异常 {skipped_bad}、样本过短 {skipped_short}")
    return panel


# ---------------------------------------------------------------------------
# IC 分析
# ---------------------------------------------------------------------------

def daily_ic(panel: pd.DataFrame, factor: str, horizon: int,
             min_n: int = MIN_N_IC) -> pd.DataFrame:
    """逐日横截面 Spearman rank IC：corr(因子, 未来 h 日收益)。"""
    fwd_col = f"fwd_{horizon}"
    rows: list[dict] = []
    for date, g in panel.groupby("date", sort=True):
        a, b = g[factor], g[fwd_col]
        mask = a.notna() & b.notna()
        n = int(mask.sum())
        if n < min_n:
            continue
        ic = a[mask].rank().corr(b[mask].rank())
        rows.append({"date": date, "ic": float(ic), "n": n})
    return pd.DataFrame(rows)


def compute_ic_matrix(panel: pd.DataFrame,
                      min_n: int = MIN_N_IC) -> dict[str, dict[int, pd.DataFrame]]:
    """单次按日遍历计算全部 因子 × 窗口 的日度 rank IC（比逐组合 groupby 快一个量级）。"""
    fwd_cols = {h: f"fwd_{h}" for h in HORIZONS}
    out: dict[str, dict[int, dict]] = {f: {h: [] for h in HORIZONS}
                                       for f in ALL_FACTORS}
    for date, g in panel.groupby("date", sort=True):
        for factor in ALL_FACTORS:
            a = g[factor]
            if a.notna().sum() < min_n:
                continue
            for h in HORIZONS:
                b = g[fwd_cols[h]]
                mask = a.notna() & b.notna()
                n = int(mask.sum())
                if n < min_n:
                    continue
                ic = float(a[mask].rank().corr(b[mask].rank()))
                out[factor][h].append({"date": date, "ic": ic, "n": n})
    return {f: {h: pd.DataFrame(out[f][h], columns=["date", "ic", "n"])
                for h in HORIZONS} for f in ALL_FACTORS}


def summarize_ic(ic_df: pd.DataFrame) -> dict:
    """IC 汇总：天数、均值、标准差、t 值、正占比、年化 ICIR（独立样本假设）。"""
    n = int(len(ic_df))
    mean = float(ic_df["ic"].mean())
    std = float(ic_df["ic"].std(ddof=1))
    t = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
    icir = mean / std * math.sqrt(252) if std > 0 else float("nan")
    return {
        "n_days": n, "mean_ic": mean, "std_ic": std, "t_ic": t,
        "pct_pos": float((ic_df["ic"] > 0).mean()), "icir_ann": icir,
    }


# ---------------------------------------------------------------------------
# 五分位组合回测
# ---------------------------------------------------------------------------

def _rebalance_dates(dates: list[pd.Timestamp], panel: pd.DataFrame,
                     min_n: int = MIN_N_REBAL) -> list[pd.Timestamp]:
    """第一个满足最少因子数的交易日起，每 5 个交易日一个调仓日。"""
    valid = panel.groupby("date").size()
    first = None
    for d in dates:
        if valid.get(d, 0) >= min_n:
            first = d
            break
    if first is None:
        return []
    return dates[dates.index(first)::REBALANCE_EVERY]


def quintile_backtest(panel_by_date: dict, factor: str,
                      dates: list[pd.Timestamp], rb_dates: list[pd.Timestamp],
                      ret_matrix: pd.DataFrame, commission: float = COMMISSION,
                      stamp: float = STAMP,
                      min_n: int = MIN_N_REBAL // 2) -> tuple[dict, dict, dict,
                                                               pd.Series,
                                                               pd.Series,
                                                               pd.Series, list]:
    """单个因子的五分位组合回测。

    panel_by_date 为 date → 当日横截面 DataFrame 的预索引字典。返回
    (top 汇总, bottom 汇总, 多空汇总, top 净值曲线, bottom 净值曲线,
    多空净值曲线, 调仓明细)。t 日收盘后按因子分位选股，t+1 日起按持仓名单
    的日收益等权计净值（停牌日收益按 0 冻结）；每次调仓的买卖权重差按
    买入佣金/卖出佣金+印花 计费，费用在调仓次日（执行日）扣减。
    """
    symbols = list(ret_matrix.columns)
    sym_index = {s: i for i, s in enumerate(symbols)}

    def leg(which: str) -> tuple[dict, pd.Series, float]:
        w_prev = np.zeros(len(symbols))
        curve: dict[pd.Timestamp, float] = {}
        nav = 1.0
        turnover_sum = 0.0
        n_held: list[int] = []
        rows: list[dict] = []
        for k, d in enumerate(rb_dates):
            f_vals = panel_by_date[d][["symbol", factor]].dropna()
            if len(f_vals) < max(min_n, 2):
                continue
            try:
                q = pd.qcut(f_vals[factor].rank(method="first"), 5,
                            labels=False, duplicates="drop")
            except ValueError:
                q = pd.qcut(f_vals[factor], 5, labels=False, duplicates="drop")
            f_vals = f_vals.assign(bin=q)
            take = ((f_vals["bin"] == q.max()) if which == "top"
                    else (f_vals["bin"] == q.min()))
            members = f_vals[take]["symbol"].tolist()
            if not members:
                continue
            w_new = np.zeros(len(symbols))
            for s in members:
                if s in sym_index:
                    w_new[sym_index[s]] = 1.0 / len(members)
            buys = float(np.maximum(w_new - w_prev, 0).sum())
            sells = float(np.maximum(w_prev - w_new, 0).sum())
            cost = buys * commission + sells * (commission + stamp)
            turnover = 0.5 * (buys + sells)
            turnover_sum += turnover
            n_held.append(len(members))
            rows.append({"date": d, "n": len(members), "turnover": turnover,
                         "cost": cost})
            w_prev = w_new

            start_i = dates.index(d) + 1
            end_i = (dates.index(rb_dates[k + 1])
                     if k + 1 < len(rb_dates) else len(dates) - 1)
            for di in range(start_i, end_i + 1):
                day = dates[di]
                r_row = (ret_matrix.loc[day].values if day in ret_matrix.index
                         else np.full(len(symbols), np.nan))
                gross = np.nansum(w_new * np.where(np.isnan(r_row), 0.0, r_row))
                nav *= 1.0 + gross - (cost if di == start_i else 0.0)
                curve[day] = nav
        series = pd.Series(curve).sort_index()
        n_reb = int(len(rows))
        return (_leg_metrics(series, turnover_sum, n_reb, n_held),
                series, turnover_sum, rows)

    top_m, top_curve, top_to, top_rows = leg("top")
    bot_m, bot_curve, bot_to, bot_rows = leg("bottom")

    # 多空：腿间日收益差（自融资假设），两腿成本各自已扣
    ls_curve = pd.Series(dtype=float)
    common = top_curve.index.intersection(bot_curve.index)
    if len(common) > 1:
        t_r = top_curve.reindex(common).pct_change().fillna(0.0)
        b_r = bot_curve.reindex(common).pct_change().fillna(0.0)
        ls_curve = (1.0 + (t_r - b_r)).cumprod()
    ls_m = _leg_metrics(ls_curve, top_to + bot_to,
                        max(len(top_rows), len(bot_rows)), ())
    return (top_m, bot_m, ls_m, top_curve, bot_curve, ls_curve,
            top_rows + bot_rows)


def _leg_metrics(curve: pd.Series, turnover_sum: float, n_reb: int,
                 n_held: list[int] | tuple) -> dict:
    """由净值曲线计算组合指标（总收益/年化/波动/夏普/最大回撤/换手）。"""
    n = len(curve)
    total = float(curve.iloc[-1] - 1.0) if n else 0.0
    rets = curve.pct_change().dropna().values if n > 1 else np.array([])
    annual = float((1.0 + total) ** (252.0 / max(n - 1, 1)) - 1.0) if n > 1 else 0.0
    vol = float(np.std(rets, ddof=1) * math.sqrt(252)) if len(rets) > 1 else float("nan")
    sharpe = annual / vol if vol and vol > 0 else float("nan")
    peak = np.maximum.accumulate(curve.values)
    mdd = float(((curve.values / peak) - 1.0).min()) if n else 0.0
    held = float(np.nanmean(n_held)) if len(n_held) else float("nan")
    return {"total": total, "annual": annual, "vol": vol, "sharpe": sharpe,
            "mdd": mdd, "bars": n, "n_reb": n_reb, "n_held_mean": held,
            "turnover_avg": turnover_sum / n_reb if n_reb else float("nan")}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _summary_rows(panel: pd.DataFrame, ic_sets: dict[str, dict[int, pd.DataFrame]],
                  quintiles: dict[str, tuple]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """汇总 ic_summary / quintile_summary / factor_stats / factor_corr。"""
    ic_rows = []
    for factor, per_h in ic_sets.items():
        for h, ic_df in per_h.items():
            if len(ic_df) == 0:
                ic_rows.append({"factor": factor, "horizon": h, "n_days": 0,
                                "mean_ic": np.nan, "std_ic": np.nan,
                                "t_ic": np.nan, "pct_pos": np.nan,
                                "icir_ann": np.nan})
                continue
            ic_rows.append({"factor": factor, "horizon": h,
                            **summarize_ic(ic_df)})
    ic_summary = pd.DataFrame(ic_rows)

    q_rows = []
    for factor, (top, bot, ls, *_rest) in quintiles.items():
        for leg, m in (("top", top), ("bottom", bot), ("ls", ls)):
            q_rows.append({"factor": factor, "portfolio": leg, **m})
    q_summary = pd.DataFrame(q_rows)

    stat_rows = []
    for f in ALL_FACTORS:
        s = panel[f].dropna()
        if s.empty:
            stat_rows.append({"factor": f, "n": 0, "mean": np.nan, "std": np.nan,
                              "min": np.nan, "max": np.nan, "autocorr1": np.nan})
            continue
        ac1 = s.autocorr(lag=1)
        stat_rows.append({"factor": f, "n": int(len(s)), "mean": float(s.mean()),
                          "std": float(s.std()), "min": float(s.min()),
                          "max": float(s.max()), "autocorr1": float(ac1) if np.isfinite(ac1) else np.nan})
    f_stats = pd.DataFrame(stat_rows)

    # 因子间横截面平均 Spearman 相关
    corr_rows = []
    pairs = [(a, b) for i, a in enumerate(ALL_FACTORS) for b in ALL_FACTORS[i + 1:]]
    for a, b in pairs:
        vals = []
        for date, g in panel.groupby("date", sort=False):
            x, y = g[a], g[b]
            m = x.notna() & y.notna()
            if m.sum() < MIN_N_IC:
                continue
            vals.append(x[m].rank().corr(y[m].rank()))
        corr_rows.append({"f1": a, "f2": b, "avg_spearman": float(np.nanmean(vals))
                          if vals else np.nan, "n_days": len(vals)})
    f_corr = pd.DataFrame(corr_rows)
    return ic_summary, q_summary, f_stats, f_corr


def _benchmark_series(bench: pd.DataFrame | None, dates: list[pd.Timestamp]) -> pd.Series:
    """沪深300 收盘价 → 对齐到研究交易日历的归一净值。"""
    if bench is None or bench.empty:
        return pd.Series(dtype=float)
    b = bench.set_index("date")["close"].sort_index()
    b = b.reindex(dates).ffill()
    first = b.first_valid_index()
    if first is None:
        return pd.Series(dtype=float)
    return b / b.loc[first]


def plot_ic_compare(ic_summary: pd.DataFrame, out_png: Path) -> None:
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    piv = ic_summary.pivot_table(index="factor", columns="horizon",
                                 values="mean_ic").reindex(ALL_FACTORS)
    t_piv = ic_summary.pivot_table(index="factor", columns="horizon",
                                   values="t_ic").reindex(ALL_FACTORS)
    x = np.arange(len(piv))
    width = 0.8 / len(HORIZONS)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(HORIZONS)))
    for i, h in enumerate(HORIZONS):
        bars = ax.bar(x + (i - (len(HORIZONS) - 1) / 2) * width, piv[h], width,
                      label=f"h={h}", color=colors[i])
        for xi, bar in zip(x, bars):
            t_val = t_piv[h].iloc[xi]
            if np.isfinite(t_val) and abs(t_val) > 1.96:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.0012 if bar.get_height() >= 0 else -0.0028),
                        "*" if 1.96 < abs(t_val) <= 3 else "**",
                        ha="center", fontsize=8, color="darkred")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel("平均 Spearman rank IC")
    ax.set_title(f"资金流因子日度 IC（vs 未来 h 日收益；* p<0.05，** p<0.01）")
    ax.legend(title="未来收益窗口", ncols=len(HORIZONS))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] IC 对比图已保存: {out_png}")


def plot_quintile_compare(equity_df: pd.DataFrame, out_png: Path) -> None:
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    picks = ["main_5", "main_10", "big_5", "big_10", "small_5", "rev20"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for factor in picks:
        if f"{factor}_top" in equity_df.columns:
            axes[0].plot(equity_df["date"], equity_df[f"{factor}_top"], lw=1.3,
                         label=f"{factor} Top多头(净)")
    axes[0].plot(equity_df["date"], equity_df["pool_ew"], lw=1.2,
                 color="tab:orange", label="全池等权")
    if "benchmark_300" in equity_df.columns:
        axes[0].plot(equity_df["date"], equity_df["benchmark_300"], lw=1.2,
                     color="tab:green", label="沪深300")
    axes[0].set_title("Top 分位多头净值（含成本）")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.3)

    for factor in picks:
        if f"{factor}_ls" in equity_df.columns:
            axes[1].plot(equity_df["date"], equity_df[f"{factor}_ls"], lw=1.3,
                         label=f"{factor} Top-Bottom")
    axes[1].axhline(1.0, color="gray", lw=0.8)
    axes[1].set_title("Top-Bottom 多空净值（假设口径，含成本）")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 五分位组合对比图已保存: {out_png}")


# ---------------------------------------------------------------------------
# run 命令
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    ff_dir = Path(args.ffdir)
    daily_dir = Path(args.dailydir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(ff_dir, daily_dir, offline=not args.bench)
    if panel.empty:
        print("[error] 面板为空：请先运行 fetch_fund_flow.py fetch 与 "
              "pool_backtest.py fetch（或 screener.py fetch）", file=sys.stderr)
        return 1
    dates = sorted(panel["date"].unique().tolist())
    print(f"[ok] 研究窗口: {dates[0].date()} ~ {dates[-1].date()}，"
          f"{len(dates)} 个交易日")

    # 收益矩阵（dates × symbols）
    ret_matrix = panel.pivot_table(index="date", columns="symbol", values="ret")
    ret_matrix = ret_matrix.reindex(dates)

    # 逐日按横截面一次遍历计算全部 因子 × 窗口 IC
    ic_sets = compute_ic_matrix(panel)
    print(f"[ok] IC 计算完成: {len(ALL_FACTORS)} 因子 × {len(HORIZONS)} 窗口")

    # 五分位组合（面板按日预索引，避免每次调仓全表扫描）
    rb_dates = _rebalance_dates(dates, panel)
    print(f"[ok] 调仓日: {len(rb_dates)} 个（每 {REBALANCE_EVERY} 个交易日），"
          f"首个 {rb_dates[0].date() if rb_dates else '-'}")
    panel_by_date = {d: g for d, g in panel.groupby("date", sort=True)}
    quintiles: dict[str, tuple] = {}
    equity_curves: dict[str, tuple] = {}
    for factor in ALL_FACTORS:
        if len(rb_dates) < 2:
            continue
        res = quintile_backtest(panel_by_date, factor, dates, rb_dates,
                                ret_matrix)
        quintiles[factor] = res
        equity_curves[factor] = res[3:6]
    print(f"[ok] 五分位组合计算完成: {len(quintiles)} 因子")

    # 汇总表
    ic_summary, q_summary, f_stats, f_corr = _summary_rows(panel, ic_sets, quintiles)
    ic_summary.to_csv(out_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    q_summary.to_csv(out_dir / "quintile_summary.csv", index=False, encoding="utf-8-sig")
    f_stats.to_csv(out_dir / "factor_stats.csv", index=False, encoding="utf-8-sig")
    f_corr.to_csv(out_dir / "factor_corr.csv", index=False, encoding="utf-8-sig")

    ic_daily_rows = []
    for factor, per_h in ic_sets.items():
        for h, ic_df in per_h.items():
            tmp = ic_df.copy()
            tmp["factor"] = factor
            tmp["horizon"] = h
            ic_daily_rows.append(tmp)
    if ic_daily_rows:
        ic_daily = pd.concat(ic_daily_rows, ignore_index=True)[
            ["factor", "horizon", "date", "ic", "n"]]
    else:
        ic_daily = pd.DataFrame(columns=["factor", "horizon", "date", "ic", "n"])
    ic_daily.to_csv(out_dir / "ic_daily.csv", index=False, encoding="utf-8-sig")

    # 净值曲线与基准
    bench = None
    bench_cache = daily_dir / "000300_benchmark.csv"
    if bench_cache.exists():
        try:
            cached = pd.read_csv(bench_cache).rename(
                columns={"日期": "date", "收盘": "close"})
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
            cached = cached.dropna().sort_values("date")
            if not cached.empty and cached["date"].max() >= dates[-1] - pd.Timedelta(days=5):
                bench = cached[["date", "close"]].reset_index(drop=True)
                print(f"[ok] 沪深300基准: 本地CSV {bench_cache}，{len(bench)} 行")
        except Exception:  # noqa: BLE001 缓存损坏视为不存在
            bench = None
    if bench is None and args.bench:
        try:
            bench, bench_src = load_benchmark(
                dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d"),
                offline=False, retries=3)
            print(f"[ok] 沪深300基准: {bench_src}，{len(bench)} 行")
        except Exception as exc:  # noqa: BLE001 基准失败不阻断研究主体
            print(f"[warn] 沪深300基准拉取失败，跳过基准列: {exc}", file=sys.stderr)
    bench_series = _benchmark_series(bench, dates)
    equity_df = pd.DataFrame({"date": dates})
    if not bench_series.empty:
        equity_df["benchmark_300"] = bench_series.reindex(dates).values
    pool_ew = (1 + ret_matrix.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    equity_df["pool_ew"] = pool_ew.values

    for factor, (top_c, bot_c, ls_c) in equity_curves.items():
        for name, curve in (("top", top_c), ("bottom", bot_c), ("ls", ls_c)):
            if not curve.empty:
                equity_df[f"{factor}_{name}"] = curve.reindex(dates).ffill().values
    equity_df.to_csv(out_dir / "quintile_equity.csv", index=False, encoding="utf-8-sig")

    if not args.no_plot:
        plot_ic_compare(ic_summary, out_dir / "ic_compare.png")
        plot_quintile_compare(equity_df, out_dir / "quintile_compare.png")

    print("\n=== IC 汇总（Top 10 by |t| @ h=5）===")
    h5 = ic_summary[ic_summary["horizon"] == 5].copy()
    h5["abs_t"] = h5["t_ic"].abs()
    print(h5.sort_values("abs_t", ascending=False).head(10)[
        ["factor", "n_days", "mean_ic", "t_ic", "icir_ann", "pct_pos"]]
        .to_string(index=False))
    print("\n=== 五分位组合（Top 多头 / 多空）===")
    if len(q_summary) == 0:
        print("（无数据：调仓日不足 2 个，请先完成数据拉取）")
    else:
        top_tab = q_summary[q_summary["portfolio"].isin(["top", "ls"])].pivot_table(
            index="factor", columns="portfolio",
            values=["total", "annual", "mdd", "turnover_avg"])
        print(top_tab.round(4).to_string())
    print(f"\n⚠ {DISCLAIMER}")
    return 0


# ---------------------------------------------------------------------------
# selftest（合成数据，不访问网络）
# ---------------------------------------------------------------------------

def cmd_selftest() -> int:
    rng = np.random.default_rng(7)
    n_stocks, n_days = 40, 90
    dates = pd.bdate_range("2026-05-01", periods=n_days)
    frames = {}
    for i in range(n_stocks):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        amount = np.full(n_days, 5e8) * (1 + rng.normal(0, 0.05, n_days))
        main_net = rng.normal(0, 5e6, n_days)
        big_net = main_net * 0.4
        super_net = main_net * 0.6
        small_net = -main_net  # 简化守恒：中单=0
        df = pd.DataFrame({
            "date": dates, "close": close, "amount": amount,
            "main_net": main_net, "big_net": big_net, "super_net": super_net,
            "small_net": small_net,
            "main_ratio": main_net / amount * 100,
            "big_ratio": big_net / amount * 100,
            "super_ratio": super_net / amount * 100,
            "small_ratio": small_net / amount * 100,
        })
        df["ret"] = df["close"].pct_change()
        for h in HORIZONS:
            df[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1.0
        for tier in TIERS:
            for n in WINDOWS:
                df[f"{tier}_{n}"] = (df[f"{tier}_net"].rolling(n).sum()
                                     / df["amount"].rolling(n).sum())
        for n in (5, 20):
            df[f"small_{n}"] = (df["small_net"].rolling(n).sum()
                                / df["amount"].rolling(n).sum())
        df["rev5"] = -(df["close"] / df["close"].shift(5) - 1.0)
        df["rev20"] = -(df["close"] / df["close"].shift(20) - 1.0)
        df["mom20"] = df["close"] / df["close"].shift(20) - 1.0
        df["vol20"] = -df["ret"].rolling(20).std(ddof=1)
        frames[f"s{i:02d}"] = df
    panel = (pd.concat(frames.values(), keys=frames.keys(), names=["symbol"])
             .reset_index(level=0).reset_index(drop=True))

    # 1) 无未来函数：t 日因子只依赖 ≤t 数据（扰动未来不改变过去因子）
    g = frames["s00"].copy()
    base = g["main_5"].copy()
    g.loc[g.index[-1], "main_net"] += 1e12
    perturbed = (g["main_net"].rolling(5).sum() / g["amount"].rolling(5).sum())
    assert base.iloc[:-5].equals(perturbed.iloc[:-5]), "未来数据不应影响历史因子"
    assert not np.isclose(base.iloc[-1], perturbed.iloc[-1]), "当期数据应影响当期因子"

    # 2) IC：构造完全前视因子（因子=未来 5 日收益），IC 应为 1
    panel["peek"] = panel["fwd_5"]
    ic_df = daily_ic(panel, "peek", 5, min_n=20)
    assert len(ic_df) > 50, f"IC 天数异常: {len(ic_df)}"
    assert ic_df["ic"].mean() > 0.999, f"完全前视因子 IC 应≈1，实际 {ic_df['ic'].mean()}"
    rev_ic = daily_ic(panel, "rev5", 5, min_n=20)
    assert len(rev_ic) and abs(rev_ic["ic"].mean()) < 1.0, "随机数据反转 IC 不应完美"
    ic_matrix = compute_ic_matrix(panel, min_n=20)
    assert len(ic_matrix["main_5"][5]) > 50 and len(ic_matrix["main_5"][1]) > 50, \
        "IC 矩阵覆盖天数异常"
    assert ic_matrix["main_5"][5]["ic"].abs().mean() < 1.0, "随机资金流因子 IC 不应接近 ±1"

    # 3) 五分位回测：peek 因子的 Top 必须跑赢 Bottom（合成数据允许前视）
    ret_matrix = panel.pivot_table(index="date", columns="symbol", values="ret")
    date_list = sorted(panel["date"].unique().tolist())
    rb = _rebalance_dates(date_list, panel, min_n=20)
    assert len(rb) >= 8, f"调仓日过少: {len(rb)}"
    panel_by_date = {d: g for d, g in panel.groupby("date", sort=True)}
    top, bot, ls, top_c, bot_c, ls_c, rows = quintile_backtest(
        panel_by_date, "peek", date_list, rb, ret_matrix, min_n=10)
    assert top["total"] > bot["total"], f"peek 因子 Top({top['total']}) 应优于 Bottom({bot['total']})"
    assert ls["total"] > 0, f"peek 因子多空应为正，实际 {ls['total']}"
    assert rows and all(0.0 <= r["turnover"] <= 1.0 for r in rows), "换手应在 [0,1]"
    assert top["n_held_mean"] > 0 and top["bars"] > 30, "持仓数与净值天数异常"
    assert len(top_c) > 30 and np.isfinite(top_c.values).all(), "净值曲线应有足够点数且有限"
    assert len(ls_c) > 30, "多空曲线应有足够点数"

    # 4) 成本：有成本净值 ≤ 无成本净值（peek 因子每次调仓必有换手）
    top0, *_rest0 = quintile_backtest(panel_by_date, "peek", date_list, rb,
                                      ret_matrix, commission=0.0, stamp=0.0,
                                      min_n=10)
    assert top["total"] <= top0["total"], "成本不应提高组合收益"

    # 5) 汇总表构造与列名
    ic_sets = {"main_5": {5: ic_df}, "rev5": {5: rev_ic}}
    quintiles = {"main_5": (top, bot, ls, top_c, bot_c, ls_c, rows),
                 "rev5": (top, bot, ls, top_c, bot_c, ls_c, rows)}
    ic_summary, q_summary, f_stats, f_corr = _summary_rows(panel, ic_sets, quintiles)
    assert len(ic_summary) == 2 and len(q_summary) == 6
    assert {"factor", "n_days", "mean_ic", "t_ic", "icir_ann"} <= set(ic_summary.columns)
    assert {"factor", "portfolio", "total", "mdd", "turnover_avg"} <= set(q_summary.columns)

    print("[ok] selftest 通过：无未来函数、完全前视因子 IC=1、Top>Bottom、"
          "多空为正、换手/成本口径、汇总表列名全部断言成立")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="大单资金流因子研究：因子构建 + 日度 IC + 五分位组合回测（issue #34）",
        epilog="示例: python scripts/fund_flow_factor.py run --bench",
    )
    parser.add_argument("--ffdir", default=FF_DIR_DEFAULT,
                        help=f"资金流 CSV 目录，默认 {FF_DIR_DEFAULT}")
    parser.add_argument("--dailydir", default=DAILY_DIR_DEFAULT,
                        help=f"日线 CSV 目录，默认 {DAILY_DIR_DEFAULT}")
    parser.add_argument("--outdir", default=OUT_DIR_DEFAULT,
                        help=f"输出目录，默认 {OUT_DIR_DEFAULT}")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="全流程：面板构建 + IC + 五分位组合")
    p_run.add_argument("--bench", action="store_true",
                       help="联网拉取沪深300基准（默认只用本地缓存）")
    p_run.add_argument("--no-plot", action="store_true", help="不生成 PNG 对比图")

    sub.add_parser("selftest", help="合成数据自检（不访问网络）")

    args = parser.parse_args(argv)
    if not args.command:
        parser.error("需要一个子命令: run / selftest")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass
    args = parse_args(argv)
    if args.command == "selftest":
        return cmd_selftest()
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
