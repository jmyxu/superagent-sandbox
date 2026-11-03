"""组合构建与仓位管理（量化学习项目第 5 步）。

把第 4 步 walk-forward 选出的**稳健参数策略**（默认 rsi period=7,oversold=25,
exit_level=70；候选 bollinger period=50,num_std=2.0,trend_period=0）从单标的
推广到 4 只标的的**组合层面**：

- 每只标的按策略信号给出 0/1 目标仓位（复用 scripts/strategies.py，与第 3 步
  同一信号语义：t 日收盘后产生目标仓位 → t+1 日开盘成交，无未来函数）；
- 组合目标权重 = 信号 × 权重方案，支持三种方案（--schemes）：
  ``ew`` 等权 / ``iv`` 波动率倒数加权 / ``rp`` 风险平价（等风险贡献 ERC）；
- **定期再平衡**（--rebalance-freq，默认 21 个交易日≈月度）：把漂移后的实际
  权重拉回目标权重；信号边沿（某标的 0↔1）发生当日也立即交易；
- 可选**回撤目标再平衡**触发（--dd-trigger，如 0.10）：组合净值自历史高点
  回撤达到阈值时提前触发一次再平衡，与定期再平衡叠加；
- 输出组合净值曲线（PNG）、每日净值与权重 CSV、交易明细 CSV、汇总 CSV，
  与**等权买入持有（EW-BH）** 和**沪深300** 对比；控制台打印总收益率、
  年化收益率、最大回撤、年化单边换手率。

执行口径（与第 3 步引擎一致处沿用，组合特有处如下）：
  1. 交易在触发日开盘价成交；买入付佣金 --commission（默认万三），
     卖出付佣金 + 印花税 --stamp（默认万五），未建模滑点；
  2. 组合按**小数股**全仓分配（学术口径；整手约束下 10 万资金无法在
     600519 上表达 25% 的权重，第 3/4 步已论证该口径的必要性）；
  3. 交易日历取 4 只标的交易日的并集；停牌日用前收盘价前向填充，
     信号与持仓在停牌期间冻结；
  4. 权重方案的波动率/协方差用最近 --vol-window（默认 60）个交易日的
     日收益率（截至触发日前一交易日收盘）估计，只使用过去信息；
  5. 组合回测必须在窗口首日按 t-1 收盘信号建立初始持仓（与单标的引擎
     "只响应窗口内仓位变化"的规则不同，此处为组合的必然要求）；
  6. 换手率口径：触发日单边换手 = 0.5 × Σ|交易金额| / 当日净值；
     年化单边换手率 = 全窗口交易日单边换手的均值 × 252。

调研来源（经典组合管理方法）：
  - 等权 1/N：DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive
    Diversification: How Inefficient is the 1/N Portfolio Strategy?",
    Review of Financial Studies 22(5), 1915-1953.
    https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901
  - 风险平价/等风险贡献：Maillard, Roncalli & Teiletche (2010), "The
    Properties of Equally Weighted Risk Contribution Portfolios",
    Journal of Portfolio Management 36(4), 60-70；
    Roncalli 综述 "Asset Return and Risk Parity"（含波动率倒数=朴素风险平价）:
    http://www.thierry-roncalli.com/download/Asset-Return-and-Risk-Parity.pdf
  - 低波动异象（波动率倒数加权的实证背景）：Baker, Bradley & Wurgler (2011),
    "Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility
    Anomaly", Financial Analysts Journal 67(1), 40-54.
  - 回撤控制：Grossman & Zhou (1993), "Optimal Investment Strategies for
    Controlling Drawdowns", Mathematical Finance 3(3), 241-276；
    Chekhlov, Uryasev & Zabarankin (2005), "Drawdown Measure in Portfolio
    Optimization", International Journal of Theoretical and Applied
    Finance 8(1), 13-58。
  - 定期再平衡的"波动收割"效应：Bouchey, Nemtchinov, Paulsen & Stein (2012),
    "Volatility Harvesting: Why Does Diversifying and Rebalancing Create
    Portfolio Growth?", Journal of Wealth Management 15(2), 26-35。

用法示例：
  # 默认：第 4 步稳健参数 rsi(7,25,70) × 三种权重方案，月度再平衡
  python scripts/portfolio.py

  # 第 4 步候选 bollinger(50,2.0,0)，指定方案与再平衡频率
  python scripts/portfolio.py --strategy bollinger --params period=50,num_std=2.0,trend_period=0

  # 纯权重方案对照（无信号，全部标的始终持仓）与回撤触发再平衡
  python scripts/portfolio.py --strategy none --schemes ew,iv,rp --dd-trigger 0.10

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/portfolio.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    TRADING_DAYS,
    _setup_cjk_font,
    compute_metrics,
    load_benchmark,
    load_stock,
    parse_params,
)
from strategies import create_strategy  # noqa: E402

DEFAULT_SYMBOLS = ["000001", "600519", "600036", "300750"]
# 第 4 步 walk-forward 选出的稳健参数（docs/optimization-log.md 6.5.3 节）
DEFAULT_STRATEGY = "rsi"
DEFAULT_PARAMS = "period=7,oversold=25,exit_level=70"
SCHEME_NAMES = {"ew": "等权", "iv": "波动率倒数", "rp": "风险平价"}
EPS = 1e-9  # 忽略的最小交易金额（相对净值）


# ---------------------------------------------------------------------------
# 权重方案
# ---------------------------------------------------------------------------

def scheme_weights(
    name: str,
    active: np.ndarray,
    returns: np.ndarray,
    vol_window: int,
    min_obs: int = 10,
) -> np.ndarray:
    """按方案计算**活跃标的**的权重向量（长度=标的数，非活跃置 0，和为 1）。

    returns: 截止触发日前一交易日的日收益率矩阵（n_days × n_stocks，NaN=无数据），
    只用最近 vol_window 行。σ 有效样本数不足 min_obs 的标的视为无效（置 0）。
    - ew: 等权 1/n；
    - iv: w ∝ 1/σ（波动率倒数；低波动标的获得更高权重）；
    - rp: 等风险贡献（ERC，Maillard-Roncalli-Teiletche 2010）：迭代
      w ← w·√(σ_p²/(n·RC_i)) 至风险贡献 w_i·(Σw)_i 相等；协方差用
      两两完整观测估计，估计失败时退化为 iv。
    """
    n = len(active)
    w = np.zeros(n)
    act = np.flatnonzero(active)
    if act.size == 0:
        return w
    if act.size == 1:
        w[act[0]] = 1.0
        return w

    sub = returns[-vol_window:, act]
    nobs = (~np.isnan(sub)).sum(axis=0)
    valid = nobs >= min_obs
    if not valid.any():
        return w
    sig = np.array([np.nanstd(sub[:, i]) if valid[i] else np.nan for i in range(len(valid))])
    sig[sig <= 0] = np.nan

    if name == "ew":
        raw = np.ones(act.size)
        raw[~valid] = 0.0
    elif name == "iv":
        raw = np.where(valid, 1.0 / sig, 0.0)
    elif name == "rp":
        raw = erc_weights(sub, valid)
    else:
        raise ValueError(f"未知权重方案 {name!r}，可选: {', '.join(sorted(SCHEME_NAMES))}")

    total = raw.sum()
    if total <= 0:
        return w
    w[act] = raw / total
    return w


def erc_weights(sub: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """等风险贡献（ERC）权重：w_i·(Σw)_i 对所有 i 相等。

    算法：Roncalli《Introduction to Risk Parity and Budgeting》中的定点迭代
    w ← w·√(σ_p² / (n·RC_i))（RC_i = w_i·(Σw)_i），从 1/n 出发，正定协方差
    下收敛于唯一的长仓 ERC 解（Maillard et al. 2010 命题 1/2）；不收敛或
    协方差估计失败时退化为波动率倒数（等权）权重。
    """
    n = sub.shape[1]
    if n <= 1:
        return np.ones(n)
    # 两两完整观测协方差（对角与配对统一用 ddof=1，与 np.cov 口径一致）；
    # 任一配对观测不足 2 天时整体退化
    cov = np.full((n, n), np.nan)
    for i in range(n):
        cov[i, i] = np.nanvar(sub[:, i], ddof=1) if valid[i] else np.nan
        for j in range(i + 1, n):
            if valid[i] and valid[j]:
                pair = sub[:, [i, j]]
                pair = pair[~np.isnan(pair).any(axis=1)]
                cov[i, j] = cov[j, i] = np.cov(pair[:, 0], pair[:, 1])[0, 1] if len(pair) >= 2 else np.nan
    if np.isnan(cov).any():
        sig = np.sqrt(np.diag(cov))
        sig[~valid] = np.nan
        raw = np.where(valid, 1.0 / sig, 0.0)
        return raw / raw.sum() if raw.sum() > 0 else raw

    w = np.ones(n) / n
    for _ in range(200):
        cov_w = cov @ w
        rc = w * cov_w  # 风险贡献
        sigma_p2 = float(w @ cov_w)
        if sigma_p2 <= 0:
            break
        w_new = w * np.sqrt(sigma_p2 / (n * rc))
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-12:
            w = w_new
            break
        w = w_new
    else:
        # 未收敛：检查风险贡献是否已足够平均，否则退化
        rc = w * (cov @ w)
        if np.max(np.abs(rc - rc.mean())) / rc.mean() > 1e-6:
            sig = np.sqrt(np.diag(cov))
            raw = np.where(valid, 1.0 / sig, 0.0)
            return raw / raw.sum() if raw.sum() > 0 else raw
    return w


# ---------------------------------------------------------------------------
# 组合回测核心
# ---------------------------------------------------------------------------

def align_series(dates: pd.Series, values: pd.Series, union: pd.DatetimeIndex) -> np.ndarray:
    """把某标的的序列对齐到并集交易日历，停牌日前向填充。"""
    s = values.groupby(dates).last()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(union).ffill().to_numpy(dtype=float)


BENCHMARK_CACHE = "000300_benchmark.csv"


def _read_benchmark_csv(path: Path) -> pd.DataFrame:
    """读取基准缓存 CSV（date/close 两列；兼容中文列名），统一转成 Timestamp/数值。"""
    df = pd.read_csv(path).rename(columns={"日期": "date", "收盘": "close"})
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def load_benchmark_cached(
    start: str | None,
    end: str,
    offline: bool,
    retries: int,
    datadir: str,
) -> tuple[pd.DataFrame, str]:
    """沪深300基准：优先读本地缓存（data/daily/000300_benchmark.csv），
    覆盖不足时经 backtest.load_benchmark 拉取并与缓存合并落盘。
    上游接口（东财）限流时单次请求可能耗时数分钟，调用方应设 socket 超时兜底。
    """
    cache_path = Path(datadir) / BENCHMARK_CACHE
    # start 是自然日，而行情从 start 之后的第一个交易日开始（如 20200101 → 2020-01-02），
    # 因此覆盖判定给 10 个自然日的宽限，避免跨年节假日导致每次都重新拉取
    need_min = (pd.Timestamp(start or "19900101") + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    if cache_path.exists():
        try:
            cached = _read_benchmark_csv(cache_path)
        except Exception:  # noqa: BLE001 缓存损坏视为不存在
            cached = pd.DataFrame(columns=["date", "close"])
            print(f"[warn] 基准缓存 {cache_path} 读取失败，将重新拉取", file=sys.stderr)
        if (
            not cached.empty
            and cached["date"].min() <= pd.Timestamp(need_min)
            and cached["date"].max() >= pd.Timestamp(end)
        ):
            return cached, f"本地CSV {cache_path}"
        if offline:
            return cached, f"本地CSV {cache_path}（覆盖不足，降级使用）"
    if offline:
        raise RuntimeError(
            f"offline 模式无覆盖完整的沪深300缓存（{cache_path}），无法联网拉取"
        )

    bench, source = load_benchmark(start, end, offline, retries)
    # load_benchmark 已返回标准化后的 date/close（两列），不再重复标准化
    bench = bench[["date", "close"]].copy()
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
    bench = bench.dropna().sort_values("date").reset_index(drop=True)
    merged = bench
    if cache_path.exists():
        try:
            old = _read_benchmark_csv(cache_path)
            merged = pd.concat([old, bench]).drop_duplicates(subset="date").sort_values("date")
        except Exception:  # noqa: BLE001 旧缓存损坏不影响本次运行
            print(f"[warn] 基准缓存 {cache_path} 读取失败，忽略并覆盖", file=sys.stderr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return bench, f"{source}（已缓存 {cache_path}）"


def run_portfolio(
    dfs: dict[str, pd.DataFrame],
    positions: dict[str, np.ndarray],
    union: pd.DatetimeIndex,
    benchmark: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """执行组合回测（全部权重方案一起跑，共用同一次数据对齐）。

    返回 dict：scheme → {equity, weights_df, trades, turnover_annual, trade_days,
    metrics…}，以及 ew_bh / bench 序列与元信息。
    """
    symbols = list(dfs)
    n = len(union)
    m = len(symbols)

    open_ = np.full((n, m), np.nan)
    close = np.full((n, m), np.nan)
    pos = np.full((n, m), np.nan)
    for j, sym in enumerate(symbols):
        df = dfs[sym]
        open_[:, j] = align_series(df["date"], df["open"], union)
        close[:, j] = align_series(df["date"], df["close"], union)
        p = pd.Series(positions[sym], index=df["date"])
        p = p[~p.index.duplicated(keep="last")]
        pos[:, j] = p.reindex(union).ffill().to_numpy(dtype=float)

    # 简单日收益率（停牌日前向填充 → 停牌期 0 收益），用于波动率/协方差估计
    safe_close = np.where(np.isnan(close), 0.0, close)
    prev_close = np.vstack([np.full((1, m), np.nan), safe_close[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = safe_close / prev_close - 1.0
    rets[np.isnan(close)] = np.nan
    rets[0] = np.nan

    start = pd.Timestamp(args.start) if args.start else union[0]
    eval_start = int(np.flatnonzero(union >= start)[0])

    # 沪深300基准：对齐到并集日历，窗口首日归一
    bench_full = benchmark.set_index("date")["close"].sort_index()
    bench_aligned = bench_full.reindex(union).ffill().bfill().to_numpy(dtype=float)
    bench_curve = bench_aligned / bench_aligned[eval_start]

    # 等权买入持有：窗口首日收盘等权买入，之后不再交易（无成本，沿用第 3 步口径）
    base_close = close[eval_start]  # 4 只标的 2020 年前均已上市，此处无 NaN
    ew_bh = np.full(n, np.nan)
    for k in range(eval_start, n):
        with np.errstate(invalid="ignore"):
            ew_bh[k] = np.nanmean(close[k] / base_close)
    ew_bh[:eval_start] = 1.0

    schemes = args.schemes.split(",")
    freq = args.rebalance_freq
    vol_window = args.vol_window
    dd_trigger = args.dd_trigger

    results: dict[str, dict] = {}
    for scheme in schemes:
        cash = float(args.capital)
        shares = np.zeros(m)
        equity = np.full(n, np.nan)
        equity[:eval_start] = cash
        weights_df = np.full((n, m + 1), np.nan)  # 每日收盘权重（含现金）
        trades: list[dict] = []
        daily_turnover = np.zeros(n)
        trade_days = 0
        rebalance_days = 0
        peak = float(args.capital)  # 截至 k-1 收盘的净值高点
        prev_peak = float(args.capital)  # 截至 k-2 收盘的净值高点

        for k in range(eval_start, n):
            value = np.where(~np.isnan(open_[k]), shares * open_[k], 0.0)
            nav_before = cash + float(value.sum())

            is_rebalance = (k - eval_start) % freq == 0
            # 回撤目标再平衡：自高点回撤**穿越**阈值时触发一次（边沿触发，
            # 避免回撤期内每个交易日都触发产生噪声交易）
            dd_hit = False
            if dd_trigger is not None and k > eval_start + 1:
                dd_now = equity[k - 1] / peak - 1.0
                dd_prev = equity[k - 2] / prev_peak - 1.0
                dd_hit = dd_now <= -dd_trigger < dd_prev
            sig_edge = False
            if k >= 2:
                for j in range(m):
                    a, b = pos[k - 1, j], pos[k - 2, j]
                    if not (np.isnan(a) or np.isnan(b)) and a != b:
                        sig_edge = True
                        break
            trigger = is_rebalance or dd_hit or sig_edge

            if trigger:
                active = pos[k - 1] == 1.0  # 只允许使用截至 t-1 收盘的信息
                w = scheme_weights(scheme, active, rets[:k], vol_window)
                target = w * nav_before
                trade_days += 1
                if is_rebalance:
                    rebalance_days += 1
                # 单边换手：0.5 × Σ|目标市值-交易前市值| / 交易前净值
                daily_turnover[k] = 0.5 * float(np.abs(target - value).sum()) / nav_before

                # 先卖后买；卖出付佣金+印花税，买入付佣金
                for j in np.flatnonzero(target < value):
                    sell_value = float(value[j] - target[j])
                    if sell_value <= EPS * nav_before:
                        continue
                    fee = sell_value * (args.commission + args.stamp)
                    shares[j] -= sell_value / open_[k, j]
                    cash += sell_value - fee
                    trades.append({
                        "date": union[k].strftime("%Y-%m-%d"), "symbol": symbols[j],
                        "action": "sell", "value": round(sell_value, 2),
                        "price": round(float(open_[k, j]), 4),
                        "shares": round(sell_value / open_[k, j], 6), "fee": round(fee, 2),
                    })
                for j in np.flatnonzero(target > value):
                    buy_value = float(target[j] - value[j])
                    if buy_value <= EPS * nav_before:
                        continue
                    fee = buy_value * args.commission
                    if buy_value + fee > cash:
                        buy_value = cash / (1 + args.commission)
                        fee = buy_value * args.commission
                    if buy_value <= 0:
                        continue
                    shares[j] += buy_value / open_[k, j]
                    cash -= buy_value + fee
                    trades.append({
                        "date": union[k].strftime("%Y-%m-%d"), "symbol": symbols[j],
                        "action": "buy", "value": round(buy_value, 2),
                        "price": round(float(open_[k, j]), 4),
                        "shares": round(buy_value / open_[k, j], 6), "fee": round(fee, 2),
                    })

            value = np.where(~np.isnan(close[k]), shares * close[k], 0.0)
            equity[k] = cash + float(value.sum())
            prev_peak = peak
            peak = max(peak, equity[k])
            weights_df[k, :m] = value / equity[k]
            weights_df[k, m] = cash / equity[k]

        eq = equity[eval_start:]
        daily = pd.Series(daily_turnover[eval_start:])
        turnover_annual = float(daily.mean() * TRADING_DAYS) if len(daily) else np.nan
        metrics = compute_metrics(equity, eval_start)

        wdf = pd.DataFrame(weights_df[eval_start:], columns=[f"w_{s}" for s in symbols] + ["w_cash"])
        wdf.insert(0, "date", union[eval_start:])
        tdf = pd.DataFrame(trades)
        results[scheme] = {
            "equity": eq,
            "metrics": metrics,
            "turnover_annual": turnover_annual,
            "trade_days": trade_days,
            "rebalance_days": rebalance_days,
            "weights_df": wdf,
            "trades_df": tdf,
        }

    return {
        "dates": union[eval_start:],
        "schemes": results,
        "ew_bh": ew_bh[eval_start:],
        "bench": bench_curve[eval_start:],
        "eval_start": eval_start,
    }


# ---------------------------------------------------------------------------
# 输出与绘图
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def plot_compare(out: dict, strategy_label: str, period_label: str, out_png: Path) -> None:
    """组合净值曲线对比图：各权重方案 vs 等权买入持有 vs 沪深300，下子图为各方案回撤。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    dates = out["dates"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    colors = {"ew": "tab:blue", "iv": "tab:red", "rp": "tab:purple"}
    for scheme, res in out["schemes"].items():
        nav = res["equity"] / res["equity"][0]
        ax1.plot(dates, nav, label=f"{SCHEME_NAMES[scheme]}（{scheme}）", lw=1.4, color=colors.get(scheme))
        dd = nav / np.maximum.accumulate(nav) - 1
        ax2.plot(dates, dd, lw=0.8, color=colors.get(scheme), alpha=0.7)
    ax1.plot(dates, out["ew_bh"], label="等权买入持有（EW-BH）", lw=1.1, alpha=0.75, color="tab:orange")
    ax1.plot(dates, out["bench"], label="沪深300基准", lw=1.1, alpha=0.75, color="tab:green")
    ax1.set_title(f"组合净值曲线 · {strategy_label}（{period_label}）")
    ax1.set_ylabel("净值（期初=1）")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("组合回撤")
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 组合净值对比图已保存: {out_png}")


def write_outputs(out: dict, strategy: object, args: argparse.Namespace, end: str) -> None:
    """落盘每日净值/权重 CSV、交易明细 CSV、对比图与汇总 CSV，并打印指标表。"""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    if strategy is None:
        label, fname = "无信号（纯权重方案）", "none"
    else:
        label, fname = strategy.label(), strategy.file_label()
    period_label = f"{args.start or '数据起点'} ~ {end}"

    for scheme, res in out["schemes"].items():
        eq_df = res["weights_df"].copy()
        eq_df["portfolio_equity"] = res["equity"] / res["equity"][0]
        eq_df["ew_bh_equity"] = out["ew_bh"]
        eq_df["benchmark_300"] = out["bench"]
        eq_df.to_csv(outdir / f"{fname}_{scheme}{suffix}_equity.csv", index=False, encoding="utf-8-sig")
        if not res["trades_df"].empty:
            res["trades_df"].to_csv(outdir / f"{fname}_{scheme}{suffix}_trades.csv", index=False, encoding="utf-8-sig")
        else:
            print(f"[warn] {scheme}: 该区间内未产生任何交易", file=sys.stderr)

    if not args.no_plot:
        plot_compare(out, label, period_label, outdir / f"{fname}{suffix}_compare.png")

    bh_metrics = compute_metrics(out["ew_bh"] * args.capital, 0)
    bench_metrics = compute_metrics(out["bench"] * args.capital, 0)

    print(f"\n===== 组合对比 · {label}（{period_label}，{args.adjust} 复权，"
          f"再平衡频率 {args.rebalance_freq} 交易日）=====")
    table: dict[str, dict[str, str]] = {}
    for scheme, res in out["schemes"].items():
        m = res["metrics"]
        table[f"{SCHEME_NAMES[scheme]}（{scheme}）"] = {
            "总收益率": _fmt_pct(m["total"]),
            "年化收益率": _fmt_pct(m["annual"]),
            "最大回撤": _fmt_pct(m["mdd"]),
            "年化单边换手": f"{res['turnover_annual'] * 100:.1f}%",
            "交易天数": str(res["trade_days"]),
        }
    table["等权买入持有"] = {
        "总收益率": _fmt_pct(bh_metrics["total"]),
        "年化收益率": _fmt_pct(bh_metrics["annual"]),
        "最大回撤": _fmt_pct(bh_metrics["mdd"]),
        "年化单边换手": "0",
        "交易天数": "0",
    }
    table["沪深300基准"] = {
        "总收益率": _fmt_pct(bench_metrics["total"]),
        "年化收益率": _fmt_pct(bench_metrics["annual"]),
        "最大回撤": _fmt_pct(bench_metrics["mdd"]),
        "年化单边换手": "—",
        "交易天数": "—",
    }
    print(pd.DataFrame(table).to_string())

    summary = []
    for scheme, res in out["schemes"].items():
        m = res["metrics"]
        summary.append({
            "series": f"{scheme}（{SCHEME_NAMES[scheme]}）", "type": "strategy",
            "total": m["total"], "annual": m["annual"], "mdd": m["mdd"],
            "turnover_annual": res["turnover_annual"],
            "trade_days": res["trade_days"], "rebalance_days": res["rebalance_days"],
        })
    summary.append({"series": "ew_bh（等权买入持有）", "type": "benchmark",
                    "total": bh_metrics["total"], "annual": bh_metrics["annual"],
                    "mdd": bh_metrics["mdd"], "turnover_annual": 0.0,
                    "trade_days": 0, "rebalance_days": 0})
    summary.append({"series": "csi300（沪深300）", "type": "benchmark",
                    "total": bench_metrics["total"], "annual": bench_metrics["annual"],
                    "mdd": bench_metrics["mdd"], "turnover_annual": np.nan,
                    "trade_days": np.nan, "rebalance_days": np.nan})
    summary_path = outdir / f"summary_{fname}{suffix}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[ok] 汇总结果已保存: {summary_path}")


# ---------------------------------------------------------------------------
# 自检与命令行入口
# ---------------------------------------------------------------------------

def self_test() -> None:
    """合成数据上的算法自检（--selftest 运行，不访问网络）。"""
    rng = np.random.default_rng(42)
    days, n = 300, 4
    # 用整周期余弦基构造"样本协方差恰好为对角阵"的收益率：列两两正交且
    # 列均值为 0（浮点误差 ~1e-14），使 ERC=波动率倒数的检验不依赖渐近近似
    k = np.arange(days)[:, None]
    freq = np.arange(1, n + 1)[None, :]
    basis = np.cos(2 * np.pi * freq * k / days)
    assert np.abs(basis.mean(axis=0)).max() < 1e-13
    gram = basis.T @ basis
    gram_off = gram.copy()
    np.fill_diagonal(gram_off, 0.0)
    assert np.abs(gram_off).max() < 1e-10 * (basis ** 2).sum()
    sig = np.array([0.01, 0.02, 0.03, 0.04])
    diag = basis * sig
    active = np.ones(n, dtype=bool)
    # 1) 对角协方差时 ERC 退化为波动率倒数
    w = scheme_weights("rp", active, diag, vol_window=days, min_obs=10)
    iv = scheme_weights("iv", active, diag, vol_window=days, min_obs=10)
    assert np.allclose(w, iv, atol=1e-8), f"对角协方差下 rp 应等于 iv: {w} vs {iv}"
    # 2) 一般协方差下 ERC 的风险贡献确实相等
    corr = rng.normal(size=(days, n))
    cov_mat = np.cov(corr.T)
    w_corr = scheme_weights("rp", active, corr, vol_window=days, min_obs=10)
    rc = w_corr * (cov_mat @ w_corr)
    assert np.max(np.abs(rc - rc.mean())) < 1e-6 * rc.mean(), f"风险贡献不等: {rc}"
    # 3) 权重和为 1、非活跃标的权重为 0
    assert abs(w.sum() - 1.0) < 1e-12
    w2 = scheme_weights("ew", np.array([True, False, True, True]), corr, days)
    assert abs(w2.sum() - 1.0) < 1e-12 and w2[1] == 0.0 and abs(w2[0] - 1 / 3) < 1e-12
    print("[ok] self_test 通过：ERC/IV/EW 权重算法自检全部断言成立")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="组合构建与仓位管理：第 4 步稳健参数策略 × 权重方案 × 定期再平衡",
        epilog=(
            "示例: python scripts/portfolio.py --strategy bollinger "
            "--params period=50,num_std=2.0,trend_period=0\n"
            "权重方案: ew 等权 / iv 波动率倒数 / rp 风险平价"
        ),
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help=f"股票代码，默认: {' '.join(DEFAULT_SYMBOLS)}")
    parser.add_argument("--start", default="20200101", help="回测开始日期 YYYYMMDD，默认 20200101")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYYMMDD，默认今天")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        help="策略名: rsi（默认，第 4 步稳健参数 7/25/70）、bollinger、"
                             "ma_cross、momentum、none（无信号纯权重方案）")
    parser.add_argument("--params", default=DEFAULT_PARAMS,
                        help=f"策略参数 k=v,k=v；默认（第 4 步稳健参数）: {DEFAULT_PARAMS}")
    parser.add_argument("--schemes", default="ew,iv,rp",
                        help="权重方案，逗号分隔: ew,iv,rp（默认全部三种）")
    parser.add_argument("--rebalance-freq", type=int, default=21,
                        help="定期再平衡频率（交易日），默认 21（约月度）")
    parser.add_argument("--vol-window", type=int, default=60,
                        help="波动率/协方差估计窗口（交易日），默认 60")
    parser.add_argument("--dd-trigger", type=float, default=None,
                        help="回撤目标再平衡阈值（如 0.10）：组合净值自高点回撤**穿越**该阈值时额外触发一次再平衡")
    parser.add_argument("--adjust", choices=["qfq", "hfq", "none"], default="qfq",
                        help="复权方式: qfq 前复权(默认) / hfq 后复权 / none 不复权")
    parser.add_argument("--datadir", default="data/daily", help="第 1 步 CSV 目录，默认 data/daily")
    parser.add_argument("--outdir", default="output/portfolio", help="输出目录，默认 output/portfolio")
    parser.add_argument("--tag", default=None, help="输出文件名后缀（多配置对照时区分，如 dd10）")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金（元），默认 100000")
    parser.add_argument("--commission", type=float, default=0.0003, help="单边佣金费率，默认 0.0003（万三）")
    parser.add_argument("--stamp", type=float, default=0.0005, help="卖出印花税费率，默认 0.0005（万五）")
    parser.add_argument("--retries", type=int, default=3, help="数据拉取失败重试次数，默认 3")
    parser.add_argument("--offline", action="store_true", help="只用本地 CSV、不访问网络")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 对比图")
    parser.add_argument("--selftest", action="store_true", help="只运行合成数据上的算法自检（不拉数据）")
    args = parser.parse_args(argv)
    if args.rebalance_freq <= 0 or args.vol_window <= 0:
        parser.error("--rebalance-freq / --vol-window 必须为正整数")
    if args.dd_trigger is not None and not (0.0 < args.dd_trigger < 1.0):
        parser.error("--dd-trigger 必须在 (0, 1) 之间")
    for s in args.schemes.split(","):
        if s not in SCHEME_NAMES:
            parser.error(f"未知权重方案 {s!r}，可选: {', '.join(sorted(SCHEME_NAMES))}")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass

    args = parse_args(argv)
    if args.selftest:
        self_test()
        return 0

    if args.strategy == "none":
        strategy = None
        warmup = args.vol_window + args.rebalance_freq + 5
        params_text = ""
    else:
        try:
            strategy = create_strategy(args.strategy, **parse_params(args.params))
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        warmup = strategy.warmup + args.vol_window + args.rebalance_freq + 5
        params_text = f"，参数 {strategy.param_string()}"

    end = args.end or datetime.now().strftime("%Y%m%d")
    print(f"回测区间: {args.start} ~ {end} | 策略: {args.strategy}{params_text} | "
          f"权重方案: {args.schemes} | 再平衡: 每 {args.rebalance_freq} 交易日"
          + (f" + 回撤触发 {args.dd_trigger:.0%}" if args.dd_trigger else "")
          + f" | 复权: {args.adjust} | 初始资金: {args.capital:,.0f}")

    benchmark, bench_source = None, None
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(90)  # 东财限流时请求可能很慢，兜底防止无限挂起
    try:
        benchmark, bench_source = load_benchmark_cached(
            args.start, end, args.offline, args.retries, args.datadir
        )
    finally:
        socket.setdefaulttimeout(old_timeout)
    print(f"[ok] 沪深300基准数据: {bench_source}，{len(benchmark)} 行")

    dfs: dict[str, pd.DataFrame] = {}
    positions: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for i, raw in enumerate(args.symbols):
        if i > 0:
            time.sleep(1.0)  # 尊重上游接口的请求间隔
        try:
            match = re.search(r"\d{6}", raw.strip())
            if not match:
                raise ValueError(f"无法识别股票代码: {raw!r}")
            symbol = match.group(0)
            df, data_source = load_stock(symbol, args.start, end, args.adjust,
                                         args.datadir, args.offline, args.retries, warmup)
            if df.empty:
                raise ValueError("无可用行情数据")
            print(f"[ok] {symbol}: 行情数据来自 {data_source}，{len(df)} 行")
            if strategy is None:
                positions[symbol] = np.ones(len(df))  # 纯权重方案：始终持仓
            else:
                positions[symbol] = strategy.positions(df)
            dfs[symbol] = df
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            print(f"[failed] {raw}: {exc}", file=sys.stderr)
            failed.append(raw)

    if len(dfs) < 2:
        print(f"[error] 可用标的不足 2 只（{len(dfs)}），无法构建组合", file=sys.stderr)
        return 1

    union = pd.DatetimeIndex(sorted(set().union(*(set(df["date"]) for df in dfs.values()))))
    union = union[union <= pd.Timestamp(end)]
    out = run_portfolio(dfs, positions, union, benchmark, args)

    write_outputs(out, strategy, args, end)

    print(f"\n⚠ {DISCLAIMER}")
    if failed:
        print(f"共 {len(args.symbols)} 只，失败 {len(failed)} 只: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
