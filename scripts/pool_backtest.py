"""全股票池样本扩展回测（RSI / 布林带 / MA 交叉，issue #31）。

动机：三个单股策略（ma_cross / rsi / bollinger）此前只在 4 只事后挑选的
幸存者标的上回测过（第 3 步口径，样本量不足）。本脚本把同一套统一回测
引擎（scripts/backtest.py 的仓位边沿成交口径）应用到第 6 步的
沪深300+中证500 股票池（约 800 只），做全样本扩展回测，并给出横截面
统计（均值/中位数、跑赢买入持有与沪深300 的比例、配对 t 统计量、符号
检验）来检验单票结论在全池上是否成立。

与 scripts/screener.py 的关系：
  - fetch/status 直接复用 screener 的股票池快照与日线拉取（增量缓存 +
    断点续跑，落盘 data/daily/<代码>_qfq.csv），两个脚本共用同一份数据；
  - backtest 阶段只读本地 CSV、不访问网络，单票结果按行写入
    summary_pool_<策略名>[_frac].csv（不在 repo 落 800×N 个逐票文件，
    逐票净值/交易明细可用 scripts/backtest.py <代码> 单独复现）。

用法示例：
  python scripts/pool_backtest.py fetch --prefer tx   # 拉取股票池日线（断点续跑）
  python scripts/pool_backtest.py status              # 拉取进度/失败清单
  python scripts/pool_backtest.py backtest            # 三策略 × 全池回测（整手+小数股）
  python scripts/pool_backtest.py backtest --strategies rsi,bollinger --lot-only
  python scripts/pool_backtest.py selftest            # 合成数据自检（不访问网络）

输出（output/backtest/pool/）：
  - summary_pool_<策略名>.csv / summary_pool_<策略名>_frac.csv：逐票一行指标
    （策略/买入持有/沪深300 三项对比、相对买入持有超额、胜率、盈亏比）；
  - pool_aggregate.csv：横截面统计（策略 × 口径 各一行，均值/中位数、
    跑赢比例、配对 t、符号检验 z）；
  - pool_yearly_<策略名>.csv：分年稳定性汇总（每年一行，完整年度样本口径：
    全池平均策略收益/买入持有/沪深300、正收益比例、跑赢买入持有比例；
    当年 IPO 的 partial 样本单列 n_partial 且不参与均值）；
  - pool_yearly_rows_<策略名>.csv：逐票×逐年明细（partial 标记，供复核）；
  - pool_mean_equity_<策略名>.csv：全池等权平均净值曲线（策略均值 vs 平均
    买入持有 vs 沪深300，逐日参与股票数 n_stocks）；
  - pool_compare.png / pool_compare_frac.png：三策略 ×（收益分布直方图 +
    全池平均净值曲线）对比图。

免责声明：历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/pool_backtest.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    _setup_cjk_font,
    compute_metrics,
    run_backtest,
    standardize_stock,
    trade_stats,
)
from screener import (  # noqa: E402
    FETCH_END_DEFAULT,
    FETCH_START_DEFAULT,
    _load_benchmark_cached,
    cmd_status,
    fetch_universe,
    run_fetch,
)
from strategies import REGISTRY, create_strategy  # noqa: E402

BACKTEST_START_DEFAULT = "20200101"  # 与项目统一窗口一致（第 0 章口径）
BACKTEST_END_DEFAULT = "20251231"
DATADIR_DEFAULT = "data/daily"
OUTDIR_DEFAULT = "output/backtest/pool"

# 三个单股策略（issue #31 指定），各自使用注册默认参数：
# ma_cross(5,20)、rsi(14,30,70)、bollinger(20,2.0,0)
POOL_STRATEGIES = ("ma_cross", "rsi", "bollinger")

AGG_COLUMNS = (
    "strat_total", "strat_annual", "strat_mdd", "strat_win_rate",
    "strat_trades", "pl_ratio", "bh_total", "bench_total",
    "excess_bh", "excess_bench",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全股票池样本扩展回测：三单股策略（RSI/布林带/MA 交叉）"
                    " × 沪深300+中证500 约 800 只（issue #31）",
        epilog=(
            "示例:\n"
            "  python scripts/pool_backtest.py fetch --prefer tx\n"
            "  python scripts/pool_backtest.py backtest\n"
            "  python scripts/pool_backtest.py selftest"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="拉取股票池成分股与日线（复用 screener，断点续跑）")
    p_fetch.add_argument("--fetch-start", default=FETCH_START_DEFAULT,
                         help=f"日线拉取起点 YYYYMMDD，默认 {FETCH_START_DEFAULT}（策略热身）")
    p_fetch.add_argument("--fetch-end", default=FETCH_END_DEFAULT,
                         help=f"日线拉取终点 YYYYMMDD，默认 {FETCH_END_DEFAULT}（项目统一窗口）")
    p_fetch.add_argument("--sleep", type=float, default=0.3,
                         help="两只股票之间的请求间隔（秒），默认 0.3，勿设 0 以免上游限流")
    p_fetch.add_argument("--retries", type=int, default=3, help="单只股票失败重试次数，默认 3")
    p_fetch.add_argument("--max-stocks", type=int, default=None,
                         help="只处理前 N 只（分批拉取用，续跑自动跳过已完成）")
    p_fetch.add_argument("--prefer", choices=["em", "tx"], default="em",
                         help="数据源优先：em 东财（默认，失败退腾讯）/ tx 腾讯（东财 IP 限流时段用）")

    sub.add_parser("status", help="查看日线拉取进度与失败清单（复用 screener）")

    p_bt = sub.add_parser("backtest", help="三策略 × 全池回测 + 横截面统计")
    p_bt.add_argument("--start", default=BACKTEST_START_DEFAULT,
                      help=f"回测开始日期 YYYYMMDD，默认 {BACKTEST_START_DEFAULT}")
    p_bt.add_argument("--end", default=BACKTEST_END_DEFAULT,
                      help=f"回测结束日期 YYYYMMDD，默认 {BACKTEST_END_DEFAULT}")
    p_bt.add_argument("--strategies", default=",".join(POOL_STRATEGIES),
                      help=f"逗号分隔的策略名（ma_cross/rsi/bollinger），默认全部三个")
    p_bt.add_argument("--datadir", default=DATADIR_DEFAULT,
                      help=f"日线 CSV 目录，默认 {DATADIR_DEFAULT}（screener fetch 落盘处）")
    p_bt.add_argument("--outdir", default=OUTDIR_DEFAULT, help=f"输出目录，默认 {OUTDIR_DEFAULT}")
    p_bt.add_argument("--capital", type=float, default=100000.0, help="初始资金（元），默认 100000")
    p_bt.add_argument("--commission", type=float, default=0.0003, help="单边佣金费率，默认 0.0003（万三）")
    p_bt.add_argument("--stamp", type=float, default=0.0005, help="卖出印花税费率，默认 0.0005（万五）")
    p_bt.add_argument("--adjust", choices=["qfq", "hfq", "none"], default="qfq", help="复权方式，默认 qfq")
    p_bt.add_argument("--retries", type=int, default=3, help="基准数据拉取失败重试次数，默认 3")
    p_bt.add_argument("--offline", action="store_true", help="只用本地 CSV 与本地股票池快照，不访问网络")
    p_bt.add_argument("--lot-only", action="store_true",
                      help="只跑整手口径（默认整手 + 小数股两种口径都跑）")
    p_bt.add_argument("--frac-only", action="store_true",
                      help="只跑小数股全仓口径（默认整手 + 小数股两种口径都跑）")
    p_bt.add_argument("--no-plot", action="store_true", help="不生成 PNG 对比图")

    sub.add_parser("selftest", help="只运行合成数据自检（不拉数据）")

    args = parser.parse_args(argv)
    if args.command == "backtest":
        names = [s.strip() for s in args.strategies.split(",") if s.strip()]
        bad = [s for s in names if s not in REGISTRY]
        if bad:
            parser.error(f"未知策略 {bad}，可选: {', '.join(sorted(REGISTRY))}")
        if not names:
            parser.error("--strategies 至少需要一个策略名")
        args.names = names
    return args


# ---------------------------------------------------------------------------
# 逐票回测（复用统一引擎，只读本地 CSV，不访问网络）
# ---------------------------------------------------------------------------

def run_stock(
    symbol: str,
    df: pd.DataFrame,
    strategy,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    capital: float,
    commission: float,
    stamp: float,
    fractional: bool,
    benchmark: pd.DataFrame,
) -> tuple[dict, pd.Series, pd.Series, list[dict]] | None:
    """对单只股票执行统一引擎回测，返回 (指标行, 策略净值曲线, 买入持有曲线, 分年收益)。

    CSV 覆盖不足（回测窗口内 0 根 K 线）返回 None（计入 skipped，不算失败）。
    """
    window = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    if not window.any():
        return None

    equity, trades, eval_start, lot_skips = run_backtest(
        df, strategy,
        start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d"),
        capital, commission, stamp, fractional,
    )
    if equity[eval_start] <= 0:  # 防御性保护，正常不可能发生
        return None

    strat_curve = pd.Series(equity / equity[eval_start], index=df["date"])
    bh_curve = pd.Series(df["close"].values / df["close"].values[eval_start],
                         index=df["date"])

    # 沪深300：对齐到个股交易日（停牌日前向填充），窗口首日归一
    bench_full = benchmark.set_index("date")["close"].sort_index()
    bench_aligned = bench_full.reindex(df["date"]).ffill().bfill()
    bench_total = np.nan
    if not bench_aligned.isna().all() and bench_aligned.iloc[eval_start] > 0:
        bench_total = bench_aligned.iloc[-1] / bench_aligned.iloc[eval_start] - 1.0

    metrics = compute_metrics(equity, eval_start)
    metrics.update(trade_stats(trades))
    bh_metrics = compute_metrics(bh_curve.values * capital, 0)

    # 盈亏比：平均盈利交易收益 / |平均亏损交易收益|（收益为 0 的交易不计入两侧）
    gains = [t.net_return for t in trades if t.net_return > 0]
    losses = [t.net_return for t in trades if t.net_return < 0]
    pl_ratio = np.nan
    if gains and losses:
        pl_ratio = float(np.mean(gains) / abs(np.mean(losses)))

    # 分年收益：每年用该股当年最后一根 K 线相对上一基点的净值变化计算。
    # 基点：2020 年为回测窗口起点（eval_start，此时净值恒为 1）；其后年份为
    # 上一年最后一根 K 线；当年内才上市的票用当年首根 K 线（partial=True，
    # 该年收益为不足一年的部分区间收益）。
    yearly: list[dict] = []
    dates = df["date"].values
    for y in range(start_ts.year, end_ts.year + 1):
        y_end = pd.Timestamp(f"{y}-12-31")
        end_mask = (dates >= start_ts) & (dates <= y_end)
        if not end_mask.any():
            continue
        end_i = int(np.flatnonzero(end_mask)[-1])
        if y == start_ts.year:
            base_i = eval_start
            partial = False
        else:
            prev_end = pd.Timestamp(f"{y - 1}-12-31")
            base_mask = (dates >= start_ts) & (dates <= prev_end)
            if base_mask.any():
                base_i = int(np.flatnonzero(base_mask)[-1])
                partial = False
            else:
                year_mask = (dates >= pd.Timestamp(f"{y}-01-01")) & (dates <= y_end)
                base_i = int(np.flatnonzero(year_mask)[0])
                partial = True
        if end_i <= base_i or strat_curve.iloc[base_i] <= 0:
            continue
        bench_ret = np.nan
        if (not bench_aligned.isna().all() and bench_aligned.iloc[base_i] > 0):
            bench_ret = bench_aligned.iloc[end_i] / bench_aligned.iloc[base_i] - 1.0
        yearly.append({
            "symbol": symbol,
            "year": y,
            "strat_ret": float(strat_curve.iloc[end_i] / strat_curve.iloc[base_i] - 1.0),
            "bh_ret": float(bh_curve.iloc[end_i] / bh_curve.iloc[base_i] - 1.0),
            "bench_ret": float(bench_ret),
            "bars": int(end_i - base_i),
            "partial": partial,
        })

    return ({
        "symbol": symbol,
        "n_rows": int(len(df)),
        "first_date": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "eval_start": df["date"].iloc[eval_start].strftime("%Y-%m-%d"),
        "strat_total": metrics["total"],
        "strat_annual": metrics["annual"],
        "strat_mdd": metrics["mdd"],
        "strat_win_rate": metrics["win_rate"],
        "strat_trades": int(metrics["trades"]),
        "wins": int(metrics["wins"]),
        "pl_ratio": pl_ratio,
        "lot_skips": int(lot_skips),
        "bh_total": bh_metrics["total"],
        "bh_annual": bh_metrics["annual"],
        "bh_mdd": bh_metrics["mdd"],
        "bench_total": bench_total,
        "excess_bh": metrics["total"] - bh_metrics["total"],
        "excess_bench": metrics["total"] - bench_total if not np.isnan(bench_total) else np.nan,
    }, strat_curve, bh_curve, yearly)


def aggregate(rows: list[dict]) -> dict:
    """逐票指标行的横截面统计：均值/中位数、跑赢比例、配对 t、符号检验 z。

    均值统计对 NaN 逐列 skipna（如 0 交易的票 win_rate=NaN 不参与胜率均值）。
    """
    df = pd.DataFrame(rows)
    n = int(len(df))
    out: dict = {"n": n, "n_traded": int((df["strat_trades"] > 0).sum())}
    for col in AGG_COLUMNS:
        out[f"mean_{col}"] = float(df[col].mean())
        out[f"median_{col}"] = float(df[col].median())
    out["pct_positive"] = float((df["strat_total"] > 0).mean())
    out["pct_beat_bh"] = float((df["excess_bh"] > 0).mean())
    out["pct_beat_bench"] = float(df["excess_bench"].dropna().gt(0).mean())
    for col in ("excess_bh", "excess_bench"):
        s = df[col].dropna()
        m = int(len(s))
        if m > 1 and s.std(ddof=1) > 0:
            out[f"t_{col}"] = float(s.mean() / (s.std(ddof=1) / math.sqrt(m)))
        else:
            out[f"t_{col}"] = float("nan")
        k = int((s > 0).sum())
        out[f"sign_z_{col}"] = float((k - m / 2.0) / math.sqrt(m / 4.0)) if m else float("nan")
    return out


# ---------------------------------------------------------------------------
# 全池平均净值曲线
# ---------------------------------------------------------------------------

def pool_mean_equity(
    curves: list[tuple[str, pd.Series, pd.Series]],
    benchmark: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """逐票净值曲线在全池交易日并集日历上等权平均。

    每只票的曲线定义在各自 CSV 的日期上（上市前/退市后无数据，不参与均值），
    因此均值权重随时间随"当日有数据"的票数变化，n_stocks 列如实记录；
    沪深300 基准对齐到并集日历并自窗口首日起归一。
    """
    union = pd.DatetimeIndex(sorted({d for _, s, _ in curves for d in s.index}))
    strat_cols, bh_cols = {}, {}
    for symbol, sc, bc in curves:
        strat_cols[symbol] = sc.reindex(union)
        bh_cols[symbol] = bc.reindex(union)
    strat_df = pd.DataFrame(strat_cols, index=union)
    bh_df = pd.DataFrame(bh_cols, index=union)

    bench_full = benchmark.set_index("date")["close"].sort_index()
    bench = bench_full.reindex(union).ffill().bfill()
    bench_norm = pd.Series(np.nan, index=union)
    bench_start = union[union >= start_ts]
    if len(bench_start):
        first = bench_start[0]
        if bench.loc[first] > 0:
            bench_norm = bench / bench.loc[first]

    out = pd.DataFrame({
        "date": union,
        "mean_strat": strat_df.mean(axis=1, skipna=True).values,
        "mean_bh": bh_df.mean(axis=1, skipna=True).values,
        "benchmark_300": bench_norm.values,
        "n_stocks": strat_df.notna().sum(axis=1).values,
    })
    return out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 对比图
# ---------------------------------------------------------------------------

def plot_pool_compare(
    strategy_rows: dict[str, list[dict]],
    mean_curves: dict[str, pd.DataFrame],
    mode_label: str,
    out_png: Path,
) -> None:
    """三策略 × 2 列对比图：左列全池策略/买入持有收益分布，右列全池平均净值曲线。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    names = list(strategy_rows)
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 3.4 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        df = pd.DataFrame(strategy_rows[name])
        ax1, ax2 = axes[r][0], axes[r][1]

        bh = df["bh_total"].dropna()
        ax1.hist(bh, bins=40, alpha=0.55, color="tab:orange",
                 label=f"买入持有（n={len(bh)}）")
        ax1.hist(df["strat_total"], bins=40, alpha=0.65, color="tab:blue",
                 label=f"策略（n={len(df)}）")
        ax1.axvline(df["strat_total"].mean(), color="tab:blue", lw=1.6, ls="--")
        ax1.axvline(bh.mean(), color="tab:orange", lw=1.6, ls="--")
        ax1.set_title(
            f"{name} · {mode_label} · 策略均值 {df['strat_total'].mean():+.1%} vs "
            f"买入持有 {bh.mean():+.1%} · 跑赢 BH {df['excess_bh'].gt(0).mean():.0%}")
        ax1.set_xlabel("总收益率（2020-01 ~ 2025-12）")
        ax1.set_ylabel("股票数")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        cur = mean_curves[name]
        ax2.plot(cur["date"], cur["mean_strat"], label=f"全池均值·{name}", lw=1.5, color="tab:blue")
        ax2.plot(cur["date"], cur["mean_bh"], label="全池均值·买入持有", lw=1.2, color="tab:orange")
        ax2.plot(cur["date"], cur["benchmark_300"], label="沪深300", lw=1.2, color="tab:green")
        ax2.set_title(f"全池等权平均净值（窗口首日=1）")
        ax2.set_ylabel("净值")
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)

    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 对比图已保存: {out_png}")


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace) -> int:
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe(offline=args.offline, retries=args.retries)
    symbols = universe["symbol"].tolist()
    names = dict(zip(universe["symbol"], universe["name"]))
    index_of = dict(zip(universe["symbol"], universe["index"]))
    print(f"[ok] 股票池: {len(symbols)} 只（快照 {universe['as_of'].iloc[0]}）")

    if args.lot_only and args.frac_only:
        print("[error] --lot-only 与 --frac-only 互斥", file=sys.stderr)
        return 2
    modes = ["lot"]
    if not args.lot_only:
        modes = ["lot", "frac"]

    benchmark, bench_source = _load_benchmark_cached(args, str(datadir))
    print(f"[ok] 沪深300基准: {bench_source}，{len(benchmark)} 行")

    skipped = 0
    aggregate_rows: list[dict] = []
    rows_by_strategy: dict[str, list[dict]] = {}
    curves_by_strategy: dict[str, list[tuple[str, pd.Series, pd.Series]]] = {}
    yearly_by_strategy: dict[str, list[dict]] = {}
    t_start = time.time()
    for name in args.names:
        strategy = create_strategy(name)
        for mode in modes:
            fractional = mode == "frac"
            suffix = "_frac" if fractional else ""
            label = f"{name}{suffix}"
            rows: list[dict] = []
            curves: list[tuple[str, pd.Series, pd.Series]] = []
            yearly_rows: list[dict] = []
            for i, symbol in enumerate(symbols):
                csv_path = datadir / f"{symbol}_{args.adjust}.csv"
                if not csv_path.exists():
                    skipped += 1
                    continue
                try:
                    df = standardize_stock(pd.read_csv(csv_path))
                    if df.empty:
                        skipped += 1
                        continue
                    res = run_stock(symbol, df, strategy, start_ts, end_ts,
                                    args.capital, args.commission, args.stamp,
                                    fractional, benchmark)
                except Exception as exc:  # noqa: BLE001 单票失败不中断全池任务
                    print(f"[failed] {symbol}: {exc}", file=sys.stderr, flush=True)
                    continue
                if res is None:
                    skipped += 1
                    continue
                row, strat_curve, bh_curve, yearly = res
                row["name"] = names.get(symbol, "")
                row["index"] = index_of.get(symbol, "")
                rows.append(row)
                curves.append((symbol, strat_curve, bh_curve))
                yearly_rows.extend(yearly)
                if (i + 1) % 100 == 0:
                    print(f"[进度] {label}: {i + 1}/{len(symbols)} 已回测",
                          flush=True)

            agg = aggregate(rows)
            agg["strategy"] = name
            agg["mode"] = mode
            aggregate_rows.append(agg)

            pd.DataFrame(rows).to_csv(
                outdir / f"summary_pool_{label}.csv", index=False, encoding="utf-8-sig")
            print(f"[ok] {label}: 有效 {len(rows)} 只（窗口内无行情跳过 {skipped} 只累计），"
                  f"汇总已保存 summary_pool_{label}.csv")

            yearly_df = pd.DataFrame(yearly_rows)
            yearly_df.to_csv(outdir / f"pool_yearly_rows_{label}.csv",
                             index=False, encoding="utf-8-sig")
            yearly_df["excess_bh"] = yearly_df["strat_ret"] - yearly_df["bh_ret"]

            def _winsorized_mean(s: pd.Series) -> float:
                """1%/99% 缩尾均值：剔除当年横截面上的极端收益（qfq 伪影/爆雷股）。"""
                lo, hi = s.quantile(0.01), s.quantile(0.99)
                return float(s.clip(lo, hi).mean())

            # 分年统计只用"完整年度"样本（partial=False）：当年 IPO 的票其当年
            # 收益是上市日~年末的部分区间收益（常含 IPO 首日大涨），会系统性
            # 抬高当年均值；partial 样本另计数，不参与均值。均值列用 1%/99%
            # 缩尾口径，中位数列用原始值——两者并存以暴露 qfq 伪影的影响。
            full_year = yearly_df[~yearly_df["partial"]]
            yearly_summary = (full_year.groupby("year", dropna=False)
                              .agg(n=("strat_ret", "size"),
                                   mean_strat_ret=("strat_ret", _winsorized_mean),
                                   median_strat_ret=("strat_ret", "median"),
                                   mean_bh_ret=("bh_ret", _winsorized_mean),
                                   median_bh_ret=("bh_ret", "median"),
                                   mean_bench_ret=("bench_ret", "mean"),
                                   pct_positive=("strat_ret", lambda s: (s > 0).mean()),
                                   pct_beat_bh=("excess_bh", lambda s: (s > 0).mean()))
                              .reset_index())
            n_partial = (yearly_df[yearly_df["partial"]].groupby("year")
                         .size().rename("n_partial"))
            yearly_summary = yearly_summary.merge(n_partial, on="year", how="left")
            yearly_summary["n_partial"] = yearly_summary["n_partial"].fillna(0).astype(int)
            yearly_summary.to_csv(outdir / f"pool_yearly_{label}.csv",
                                  index=False, encoding="utf-8-sig")
            print(f"[ok] {label}: 分年汇总已保存 pool_yearly_{label}.csv"
                  f"（完整年度样本口径，当年 IPO 的 partial 样本单列）")
            yearly_by_strategy[label] = yearly_summary

            rows_by_strategy[label] = rows
            if not curves:
                print(f"[warn] {label}: 无有效净值曲线，跳过全池均值输出", file=sys.stderr)
            else:
                mean_curve = pool_mean_equity(curves, benchmark, start_ts, end_ts)
                mean_curve.to_csv(outdir / f"pool_mean_equity_{label}.csv",
                                  index=False, encoding="utf-8-sig")
                curves_by_strategy[label] = mean_curve
                print(f"[ok] {label}: 全池平均净值已保存 pool_mean_equity_{label}.csv")
            skipped = 0

    pd.DataFrame(aggregate_rows).to_csv(
        outdir / "pool_aggregate.csv", index=False, encoding="utf-8-sig")
    print(f"\n[ok] 横截面统计已保存: {outdir / 'pool_aggregate.csv'}")

    print(pd.DataFrame(aggregate_rows)[
        ["strategy", "mode", "n", "n_traded",
         "mean_strat_total", "median_strat_total", "mean_excess_bh",
         "pct_beat_bh", "t_excess_bh", "mean_excess_bench", "pct_beat_bench",
         "t_excess_bench", "mean_pl_ratio", "mean_strat_win_rate"]].to_string(index=False))

    if not args.no_plot:
        for mode in modes:
            suffix = "_frac" if mode == "frac" else ""
            labels = [f"{n}{suffix}" for n in args.names
                      if f"{n}{suffix}" in curves_by_strategy]
            if not labels:
                continue
            plot_pool_compare(
                {k: rows_by_strategy[k] for k in labels},
                {k: curves_by_strategy[k] for k in labels},
                "小数股全仓（学术口径）" if mode == "frac" else "整手（1手=100股）",
                outdir / f"pool_compare{suffix}.png",
            )

    el = time.time() - t_start
    print(f"\n全池回测完成，用时 {el:.0f}s")
    print(f"\n⚠ {DISCLAIMER}")
    return 0


def cmd_selftest() -> int:
    """合成数据自检：逐票回测、横截面统计与全池均值曲线的核心断言。"""
    from strategies import create_strategy

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    bench = pd.DataFrame({"date": dates, "close": np.linspace(3000, 4000, len(dates))})

    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("s0", "s1", "s2"):
        close = 10 + np.cumsum(rng.normal(0.001, 0.02, len(dates)))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        frames[symbol] = pd.DataFrame(
            {"date": dates, "open": np.maximum(open_, 0.01), "close": np.maximum(close, 0.01)})

    strat = create_strategy("ma_cross")
    start_ts, end_ts = pd.Timestamp("20200101"), pd.Timestamp("20251231")
    rows, curves, yearly_rows = [], [], []
    for symbol, df in frames.items():
        res = run_stock(symbol, df, strat, start_ts, end_ts, 100000.0,
                        0.0003, 0.0005, False, bench)
        assert res is not None, f"{symbol} 应产生有效回测行"
        rows.append(res[0])
        curves.append((symbol, res[1], res[2]))
        yearly_rows.extend(res[3])
    assert len(rows) == 3, f"应有 3 行，实际 {len(rows)}"

    # 分年收益：2020-2025 每年都应有一条（合成数据无缺 K 线）
    years = {r["year"] for r in yearly_rows}
    assert years == {2020, 2021, 2022, 2023, 2024, 2025}, f"分年覆盖异常: {years}"
    y2020 = [r for r in yearly_rows if r["year"] == 2020]
    assert len(y2020) == 3 and all(not r["partial"] for r in y2020), \
        "2020 年不应标记 partial"

    # 回测窗口内 0 根 K 线的票应被跳过
    empty = pd.DataFrame({"date": [pd.Timestamp("2019-01-02")], "open": [1.0], "close": [1.0]})
    assert run_stock("sx", empty, strat, start_ts, end_ts,
                     100000.0, 0.0003, 0.0005, False, bench) is None

    # 横截面统计：手算对照（两行：超额 -0.05 / +0.05）
    two = [
        {**rows[0], "excess_bh": -0.05, "excess_bench": -0.05, "strat_total": 0.10,
         "bh_total": 0.15, "bench_total": 0.05, "strat_trades": 2,
         "strat_win_rate": 0.5, "strat_annual": 0.01, "strat_mdd": -0.1,
         "pl_ratio": 1.5},
        {**rows[1], "excess_bh": 0.05, "excess_bench": 0.05, "strat_total": 0.20,
         "bh_total": 0.15, "bench_total": 0.05, "strat_trades": 0,
         "strat_win_rate": np.nan, "strat_annual": 0.02, "strat_mdd": -0.2,
         "pl_ratio": np.nan},
    ]
    agg = aggregate(two)
    assert agg["n"] == 2 and agg["n_traded"] == 1
    assert abs(agg["mean_excess_bh"]) < 1e-12, f"两行超额均值应为 0，实际 {agg['mean_excess_bh']}"
    assert abs(agg["mean_strat_win_rate"] - 0.5) < 1e-12, "0 交易票的 NaN 不应参与胜率均值"
    assert abs(agg["pct_beat_bh"] - 0.5) < 1e-12, f"跑赢比例应为 0.5，实际 {agg['pct_beat_bh']}"
    assert agg["sign_z_excess_bh"] == 0.0, "符号检验 z 应为 0（一正一负）"
    assert abs(agg["mean_pl_ratio"] - 1.5) < 1e-12, "0 交易票的盈亏比 NaN 不应参与均值"

    # 全池均值曲线：并集日历、等权、n_stocks 记录
    mean = pool_mean_equity(curves, bench, start_ts, end_ts)
    assert len(mean) > 1000, f"均值曲线长度异常: {len(mean)}"
    assert mean["n_stocks"].max() == 3, f"n_stocks 最大应为 3，实际 {mean['n_stocks'].max()}"
    assert mean["mean_strat"].iloc[0] == 1.0, "窗口首日策略均值应为 1（未建仓前为现金）"

    # 整手/小数股两种口径都应能产出（rsi 有 NaN 热身段，覆盖 NaN 路径）
    rsi = create_strategy("rsi")
    res_frac = run_stock("s0", frames["s0"], rsi, start_ts, end_ts,
                         100000.0, 0.0003, 0.0005, True, bench)
    assert res_frac is not None and res_frac[0]["strat_total"] == res_frac[0]["strat_total"], \
        "小数股口径应产出有效指标"

    print("[ok] selftest 通过：逐票回测、跳过逻辑、横截面统计（均值/胜率/盈亏比跳过 NaN、"
          "符号检验）、分年收益、全池均值曲线与整手/小数股口径断言全部成立")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass

    args = parse_args(argv)
    command = args.command or "backtest"

    if command == "selftest":
        return cmd_selftest()
    if command == "status":
        return cmd_status()
    if command == "fetch":
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(90)  # 上游限流时请求可能很慢，兜底防挂起（与 screener 一致）
        try:
            universe = fetch_universe(offline=False)
            return run_fetch(universe, args.fetch_start, args.fetch_end,
                             DATADIR_DEFAULT, args.sleep, args.retries,
                             args.max_stocks, offline=False, prefer=args.prefer)
        finally:
            socket.setdefaulttimeout(old_timeout)
    if command == "backtest":
        return cmd_backtest(args)
    print("[error] 未知命令", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
