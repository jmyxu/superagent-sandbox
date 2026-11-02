"""可插拔策略回测引擎（量化学习项目第 3 步）。

策略与引擎分离：
  - 策略定义在 scripts/strategies.py：Strategy 基类（参数注册 + 信号生成），
    内置 ma_cross（第 2 步双均线）、rsi（RSI 超卖反转）、bollinger（布林带
    均值回归）、momentum（绝对动量/动量轮动到现金），各注明参考来源链接；
  - 本文件是统一回测引擎：数据加载 → 策略信号 → 按仓位变化成交 →
    统一报告输出（指标对比表、交易明细 CSV、每日净值 CSV、收益曲线 PNG、
    汇总 CSV），任何策略共用同一套输出格式。

执行规则（统一、可复现）：
  1. 信号语义：策略在 t 日收盘后给出目标仓位 positions[t]（0=空仓持现金，
     1=全仓做多）；引擎只在仓位**发生变化**时交易（边沿触发），
     t 日的仓位变化在 t+1 日以开盘价成交——避免"当日信号当日收盘成交"的未来函数；
  2. 回测窗口 [start, end] 内只响应窗口内产生的仓位变化；窗口开始前已持有的
     状态不继承，需要新的边沿信号才会入场；
  3. 买入单位为 1 手 = 100 股；买入付佣金 --commission（默认万三），
     卖出付佣金 + 印花税 --stamp（默认万五）；
     加 --fractional 可改用小数股数全仓（学术回测口径，研究信号本身时用）；
  4. 回测期末若仍持仓，按最后收盘价强制平仓并计入交易明细与胜率；
  5. 策略所需的热身数据（如均线窗口）由策略的 warmup 属性声明，
     数据加载时自动把拉取起点前移。

数据来源：
  - 优先读取第 1 步 scripts/fetch_daily.py 产出的 CSV（data/daily/<代码>_<复权方式>.csv）；
  - CSV 缺失或覆盖不足时，按以下顺序自动拉取（东财→腾讯→新浪），并落盘为同名 CSV：
    1. 直接调用 fetch_daily（akshare 东财接口，与第 1 步同源）；
    2. 东财接口被限流/封禁时，切换 akshare 腾讯日线接口（同样支持 qfq/hfq 复权）；
    3. 腾讯接口失败时，切换 akshare 新浪日线接口（同样支持 qfq/hfq 复权）；
  - 基准为沪深 300 指数（000300）同期走势，东财接口失败时自动切换新浪备用源。
    注意：不同数据源的复权口径可能存在细微差异，跨源比对结果时需留意。

输出（统一报告）：
  - 控制台打印：总收益率、年化收益率、最大回撤、胜率、交易次数
    （策略 vs 买入持有 vs 沪深300 三列对比）；
  - output/backtest/ 下按 <代码>_<策略标签> 保存：交易明细 CSV、每日净值 CSV、
    收益曲线图 PNG（含回撤子图）；多标的运行另存 summary_<策略名>.csv 汇总；
  - 报告与图表均明确标注：历史模拟回测不代表未来，不构成投资建议。

用法示例：
  # 第 2 步双均线（默认策略，--short/--long 继续可用）
  python scripts/backtest.py 000001 600519 --start 20200101 --end 20251231 --short 5 --long 20

  # 任一注册策略 + 任意注册参数（k=v,k=v）
  python scripts/backtest.py 000001 --strategy rsi --params period=14,oversold=30,exit_level=70
  python scripts/backtest.py 000001 --strategy bollinger --params period=20,num_std=2.0
  python scripts/backtest.py 000001 --strategy momentum --params lookback=120,skip=5,threshold=0.0
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/backtest.py` 方式运行时能 import 同目录的 fetch_daily
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from strategies import Strategy, create_strategy, list_strategies  # noqa: E402

LOT_SIZE = 100  # A 股买入单位：1 手 = 100 股
TRADING_DAYS = 252  # 年化收益率使用的年交易日数
BENCHMARK_SYMBOL = "000300"  # 沪深 300
HISTORY_START = "19900101"  # 沪深 300 基准的历史拉取起点（早于其 2005 年发布日即可）

DISCLAIMER = "历史模拟回测结果不代表未来表现，不构成投资建议。"


@dataclass
class Trade:
    """一次完整的开仓→平仓往返交易，价格均为已含交易成本的净价。"""

    entry_date: pd.Timestamp
    entry_price: float  # 买入净价（含佣金）
    exit_date: pd.Timestamp
    exit_price: float  # 卖出净价（已扣佣金+印花税）
    shares: int
    net_return: float  # 本次往返净收益率
    holding_days: int  # 持有交易日数
    exit_reason: str  # "死叉卖出" 或 "期末强平"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可插拔策略回测引擎（沪深300基准对比），统一输出收益指标、交易明细与收益曲线图",
        epilog=(
            "示例: python scripts/backtest.py 000001 --strategy rsi --params period=14,oversold=30\n"
            "内置策略与默认参数:\n" + list_strategies()
        ),
    )
    parser.add_argument("symbols", nargs="+", help="股票代码，可传多个，如 000001 600519")
    parser.add_argument("--start", default=None, help="回测开始日期 YYYYMMDD，默认取数据起点（策略热身完成之后）")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYYMMDD，默认今天")
    parser.add_argument("--strategy", default="ma_cross",
                        help="策略名: ma_cross/ma（第2步双均线，默认）、rsi、bollinger、momentum")
    parser.add_argument("--params", default=None,
                        help="策略参数覆盖，逗号分隔的 k=v 对，如 period=14,oversold=30；"
                             "仅接受策略注册过的参数")
    parser.add_argument("--short", type=int, default=None,
                        help="[仅 ma_cross] 短期均线窗口（交易日），默认取策略默认值 5")
    parser.add_argument("--long", type=int, default=None,
                        help="[仅 ma_cross] 长期均线窗口（交易日），默认取策略默认值 20")
    parser.add_argument("--adjust", choices=["qfq", "hfq", "none"], default="qfq",
                        help="复权方式: qfq 前复权(默认) / hfq 后复权 / none 不复权")
    parser.add_argument("--datadir", default="data/daily", help="第 1 步 CSV 目录，默认 data/daily")
    parser.add_argument("--outdir", default="output/backtest", help="回测输出目录，默认 output/backtest")
    parser.add_argument("--tag", default=None, help="输出文件名后缀（参数敏感性批量跑时区分不同配置）")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金（元），默认 100000")
    parser.add_argument("--commission", type=float, default=0.0003, help="单边佣金费率，默认 0.0003（万三）")
    parser.add_argument("--stamp", type=float, default=0.0005, help="卖出印花税费率，默认 0.0005（万五）")
    parser.add_argument("--fractional", action="store_true",
                        help="允许不足一手的小数股数全仓买入（学术回测口径；默认按 A 股 1 手=100 股取整）")
    parser.add_argument("--retries", type=int, default=3, help="数据拉取失败重试次数，默认 3")
    parser.add_argument("--offline", action="store_true",
                        help="只用本地 CSV、不访问网络；CSV 覆盖不足时降级使用并告警（无基准数据则报错）")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 图（参数敏感性批量跑时用）")
    args = parser.parse_args(argv)
    if args.short is not None and args.short <= 0:
        parser.error("--short 必须为正整数")
    if args.long is not None and args.long <= 0:
        parser.error("--long 必须为正整数")
    if args.capital <= 0:
        parser.error("--capital 必须为正数")
    return args


def parse_params(text: str) -> dict[str, str]:
    """把 'period=14,oversold=30.5' 解析为 {k: v}（v 保持字符串，由策略做类型转换与校验）。"""
    out: dict[str, str] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--params 应为 k=v 形式，收到: {piece!r}")
        key, value = piece.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or value == "":
            raise ValueError(f"--params 键值不能为空: {piece!r}")
        out[key] = value
    return out


def build_strategy(args: argparse.Namespace) -> Strategy:
    """按 CLI 参数构造策略：--params 覆盖注册参数；--short/--long 是 ma_cross 的兼容入口。"""
    params: dict[str, str] = {}
    if args.params:
        params.update(parse_params(args.params))
    if args.strategy in ("ma_cross", "ma"):
        if args.short is not None and "short" not in params:
            params["short"] = str(args.short)
        if args.long is not None and "long" not in params:
            params["long"] = str(args.long)
    elif args.short is not None or args.long is not None:
        print("[warn] --short/--long 仅适用于 ma_cross 策略，已忽略", file=sys.stderr)
    return create_strategy(args.strategy, **params)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def standardize_stock(df: pd.DataFrame) -> pd.DataFrame:
    """把 akshare 返回的日线（中文列名）归一为 date/open/close 三列，去重排序。"""
    rename = {}
    for src, dst in (("日期", "date"), ("开盘", "open"), ("收盘", "close")):
        if src in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename)
    for col in ("date", "open", "close"):
        if col not in df.columns:
            raise ValueError(f"数据缺少列 {col!r}，实际列: {list(df.columns)}")
    df = df[["date", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    return df.reset_index(drop=True)


def exchange_prefix(symbol: str) -> str:
    """给 6 位代码加交易所前缀，供腾讯/新浪接口使用。"""
    if symbol.startswith(("6", "9", "5")):
        return "sh" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return "sz" + symbol


def load_stock(
    symbol: str,
    start: str | None,
    end: str,
    adjust: str,
    datadir: str,
    offline: bool,
    retries: int,
    warmup: int,
) -> tuple[pd.DataFrame, str]:
    """读取第 1 步产出的 CSV；覆盖不足时按 东财→腾讯→新浪 顺序拉取并落盘。

    返回 (日线数据, 数据来源说明)。warmup 为策略热身所需的最少交易日数，
    拉取起点会自动前移预留。
    """
    csv_path = Path(datadir) / f"{symbol}_{adjust}.csv"
    if csv_path.exists():
        df = standardize_stock(pd.read_csv(csv_path))
        covered = df["date"].max() >= pd.Timestamp(end) and (
            start is None or df["date"].min() <= pd.Timestamp(start)
        )
        if covered:
            return df, f"本地CSV {csv_path}"
        if offline:
            print(
                f"[warn] {symbol}: 本地 CSV 未覆盖回测区间（可能缺均线热身数据），offline 模式下降级使用",
                file=sys.stderr,
            )
            return df, f"本地CSV {csv_path}（覆盖不足，降级使用）"
    if offline:
        raise RuntimeError(
            f"{symbol}: 无覆盖完整的本地 CSV（{csv_path}），offline 模式无法联网拉取"
        )

    warm_days = int(warmup * 1.6) + 10  # 交易日约占自然日 70%，预留策略热身数据
    warm_start = None
    if start is not None:
        warm_start = (datetime.strptime(start, "%Y%m%d") - timedelta(days=warm_days)).strftime("%Y%m%d")

    errors: list[str] = []
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 保证落盘目录存在（与 fetch_daily 行为一致）
    # 数据源 1：第 1 步同源（akshare 东财），自动带热身起始日
    try:
        import fetch_daily

        raw = fetch_daily.fetch_one(symbol, warm_start, end, adjust, retries)
        df = standardize_stock(raw)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df, "akshare 东财(经 fetch_daily)"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"东财: {exc}")

    tx_adjust = "" if adjust == "none" else adjust
    prefixed = exchange_prefix(symbol)
    # 数据源 2：腾讯日线（支持 qfq/hfq）
    try:
        import akshare as ak

        raw = ak.stock_zh_a_hist_tx(
            symbol=prefixed, start_date=warm_start or HISTORY_START, end_date=end, adjust=tx_adjust
        )
        if raw is None or raw.empty:
            raise ValueError("返回空数据")
        df = standardize_stock(raw)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df, "akshare 腾讯(东财不可用)"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"腾讯: {exc}")

    # 数据源 3：新浪日线（支持 qfq/hfq）
    try:
        import akshare as ak

        raw = ak.stock_zh_a_daily(
            symbol=prefixed, start_date=warm_start or HISTORY_START, end_date=end, adjust=tx_adjust
        )
        if raw is None or raw.empty:
            raise ValueError("返回空数据")
        df = standardize_stock(raw)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return df, "akshare 新浪(东财/腾讯不可用)"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"新浪: {exc}")

    raise RuntimeError(
        f"{symbol}: 所有数据源拉取失败（{'；'.join(errors)}）。"
        "请稍后重试，或先用第 1 步脚本拉好 CSV 后加 --offline。"
    )


def load_benchmark(
    start: str | None,
    end: str,
    offline: bool,
    retries: int = 3,
) -> tuple[pd.DataFrame, str]:
    """拉取沪深300日线（东财接口优先，失败切换新浪备用源），返回 (date/close 数据, 数据源名)。"""
    if offline:
        raise RuntimeError("offline 模式无法拉取沪深300基准数据，请去掉 --offline")
    start_arg = start or HISTORY_START

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            import akshare as ak

            df = ak.index_zh_a_hist(
                symbol=BENCHMARK_SYMBOL, period="daily", start_date=start_arg, end_date=end
            )
            if df is None or df.empty:
                raise ValueError("接口返回空数据")
            return standardize_stock(df)[["date", "close"]], "akshare.index_zh_a_hist(东财)"
        except Exception as exc:  # noqa: BLE001 第三方接口异常种类多，统一重试/降级
            last_err = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            raise ValueError("新浪备用源返回空数据")
        df = standardize_stock(df)[["date", "close"]]
        df = df[(df["date"] >= pd.Timestamp(start_arg)) & (df["date"] <= pd.Timestamp(end))]
        return df.reset_index(drop=True), "akshare.stock_zh_index_daily(新浪)"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"沪深300基准数据拉取失败（东财: {last_err}；新浪: {exc}）。"
            "请稍后重试，或确认网络与 akshare 可用。"
        ) from exc


# ---------------------------------------------------------------------------
# 回测核心
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    start: str | None,
    end: str | None,
    capital: float,
    commission: float,
    stamp: float,
    fractional: bool = False,
) -> tuple[np.ndarray, list[Trade], int, int]:
    """执行策略回测，返回 (每日净值数组, 交易列表, 回测窗口起始下标, 因整手约束被跳过的信号数)。

    仓位边沿触发：策略在 t-1 日收盘后给出目标仓位（相对 t-2 日发生变化），
    则 t 日开盘成交；信号在全量数据（含热身段）上计算，但只在 [start, end]
    窗口内允许交易，且仓位变化必须发生在窗口内——避免"回测区间开头无信号
    偏差"与未来函数。
    """
    positions = strategy.positions(df)
    pos_valid = ~np.isnan(positions)
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)

    in_window = np.ones(n, dtype=bool)
    if start is not None:
        in_window &= df["date"].values >= pd.Timestamp(start)
    if end is not None:
        in_window &= df["date"].values <= pd.Timestamp(end)

    cash = float(capital)
    shares: float = 0.0
    entry_i: int | None = None
    entry_px = 0.0
    equity = np.empty(n)
    trades: list[Trade] = []
    lot_skips = 0

    for t in range(n):
        if t >= 2 and in_window[t] and in_window[t - 1] and pos_valid[t - 1] and pos_valid[t - 2]:
            prev, older = positions[t - 1], positions[t - 2]
            if shares == 0 and prev == 1 and older == 0:
                px = open_[t] * (1 + commission)
                if fractional:
                    shares = cash / px
                else:
                    lots = int((cash // px) // LOT_SIZE)
                    shares = float(lots * LOT_SIZE)
                if shares > 0:
                    cash -= shares * px
                    entry_i, entry_px = t, px
                else:
                    lot_skips += 1  # 信号出现但资金不足一手（高股价+小资金），无法买入
            elif shares > 0 and prev == 0 and older == 1:
                px = open_[t] * (1 - commission - stamp)
                trades.append(Trade(
                    entry_date=df["date"].iloc[entry_i],
                    entry_price=entry_px,
                    exit_date=df["date"].iloc[t],
                    exit_price=px,
                    shares=shares,
                    net_return=px / entry_px - 1,
                    holding_days=t - entry_i,
                    exit_reason="仓位变空卖出",
                ))
                cash += shares * px
                shares = 0
                entry_i = None
        equity[t] = cash + shares * close[t]

    # 期末仍持仓 → 按最后收盘价强平，计入交易明细与胜率
    if shares > 0:
        px = close[-1] * (1 - commission - stamp)
        trades.append(Trade(
            entry_date=df["date"].iloc[entry_i],
            entry_price=entry_px,
            exit_date=df["date"].iloc[-1],
            exit_price=px,
            shares=shares,
            net_return=px / entry_px - 1,
            holding_days=n - 1 - entry_i,
            exit_reason="期末强平",
        ))
        equity[-1] = cash + shares * px

    # 回测窗口起点：显式指定 start 用其第一个交易日；否则用策略首个信号就绪的交易日
    if start is not None:
        idx = np.flatnonzero(df["date"].values >= pd.Timestamp(start))
        eval_start = int(idx[0]) if len(idx) else n - 1
    else:
        idx = np.flatnonzero(pos_valid)
        eval_start = int(idx[0]) if len(idx) else n - 1
    return equity, trades, eval_start, lot_skips


def compute_metrics(equity: np.ndarray, start_i: int) -> dict[str, float]:
    """在 [start_i, 末尾] 窗口上计算收益指标。equity 为净值序列。"""
    eq = pd.Series(equity[start_i:]).reset_index(drop=True)
    if eq.iloc[0] <= 0:
        return {"total": np.nan, "annual": np.nan, "mdd": np.nan, "win_rate": np.nan, "trades": 0, "wins": 0}
    n_bars = len(eq)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    annual = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n_bars) - 1 if n_bars > 1 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": total, "annual": annual, "mdd": mdd, "win_rate": np.nan, "trades": 0, "wins": 0}


def trade_stats(trades: list[Trade]) -> dict[str, float]:
    wins = sum(1 for t in trades if t.net_return > 0)
    return {
        "trades": float(len(trades)),
        "wins": float(wins),
        "win_rate": wins / len(trades) if trades else np.nan,
    }


# ---------------------------------------------------------------------------
# 输出与绘图
# ---------------------------------------------------------------------------

def _setup_cjk_font() -> None:
    """注册 Windows 常见中文字体，保证图表中文标签正常显示。"""
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("未安装 matplotlib，请先执行: pip install matplotlib") from exc

    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
        font_path = Path("C:/Windows/Fonts") / name
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:  # noqa: BLE001 字体已注册时报错可忽略
                pass
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_equity_curve(
    dates: pd.Series,
    strat: np.ndarray,
    bh: np.ndarray,
    bench_dates: pd.Series,
    bench: np.ndarray,
    symbol: str,
    label: str,
    period_label: str,
    out_png: Path,
) -> None:
    """收益曲线图：上半部分三线对比，下半部分策略回撤，图底标注免责声明。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(dates, strat, label=label, lw=1.5, color="tab:blue")
    ax1.plot(dates, bh, label="买入持有（同期）", lw=1.0, alpha=0.7, color="tab:orange")
    ax1.plot(bench_dates, bench, label="沪深300基准", lw=1.0, alpha=0.7, color="tab:green")
    ax1.set_title(f"{symbol} · {label}（{period_label}）")
    ax1.set_ylabel("净值（期初=1）")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    dd = strat / np.maximum.accumulate(strat) - 1
    ax2.fill_between(dates, dd, 0, color="tab:red", alpha=0.35)
    ax2.set_ylabel("策略回撤")
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 收益曲线图已保存: {out_png}")


def _fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def run_one(
    symbol: str,
    args: argparse.Namespace,
    strategy: Strategy,
    benchmark: pd.DataFrame,
    bench_source: str,
    results: list[dict],
) -> bool:
    end = args.end or datetime.now().strftime("%Y%m%d")
    df, data_source = load_stock(symbol, args.start, end, args.adjust, args.datadir,
                                 args.offline, args.retries, strategy.warmup)
    if df.empty:
        print(f"[failed] {symbol}: 无可用行情数据", file=sys.stderr)
        return False
    print(f"[ok] {symbol}: 行情数据来自 {data_source}，{len(df)} 行")

    equity, trades, eval_start, lot_skips = run_backtest(
        df, strategy, args.start, args.end,
        args.capital, args.commission, args.stamp, args.fractional,
    )
    if lot_skips:
        print(f"[warn] {symbol}: 有 {lot_skips} 次买入信号因资金不足一手而未能买入"
              f"（当前资金 {args.capital:,.0f} 元）；加 --fractional 可用小数股数全仓研究信号本身",
              file=sys.stderr)
    dates = df["date"]
    window_dates = dates.iloc[eval_start:]

    # 买入持有基准：窗口首日收盘买入（无成本）
    bh = (df["close"].values / df["close"].values[eval_start])
    # 沪深300基准：对齐到个股交易日（停牌日用最近收盘前向填充），窗口首日归一
    bench_full = benchmark.set_index("date")["close"].sort_index()
    bench_aligned = bench_full.reindex(dates).ffill().bfill()
    if bench_aligned.isna().all():
        print(f"[failed] {symbol}: 基准数据与个股交易日无重叠", file=sys.stderr)
        return False
    bench_curve = (bench_aligned.values / bench_aligned.values[eval_start])

    strat = equity / equity[eval_start]
    metrics = compute_metrics(equity, eval_start)
    metrics.update(trade_stats(trades))

    period_label = f"{args.start or '数据起点'} ~ {end}"
    suffix = f"_{args.tag}" if args.tag else ""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label = strategy.label()
    fname = strategy.file_label()

    # 每日净值 CSV（含基准）—— 统一报告格式
    pd.DataFrame({
        "date": dates,
        "close": df["close"].values,
        "strategy_equity": strat,
        "buy_hold": bh,
        "benchmark_300": bench_curve,
    }).to_csv(outdir / f"{symbol}_{fname}{suffix}_equity.csv",
              index=False, encoding="utf-8-sig")

    # 交易明细 CSV —— 统一报告格式
    if trades:
        pd.DataFrame([{
            "entry_date": t.entry_date.strftime("%Y-%m-%d"),
            "entry_price_net": round(t.entry_price, 4),
            "exit_date": t.exit_date.strftime("%Y-%m-%d"),
            "exit_price_net": round(t.exit_price, 4),
            "shares": round(t.shares, 6),
            "net_return": round(t.net_return, 6),
            "holding_days": t.holding_days,
            "exit_reason": t.exit_reason,
        } for t in trades]).to_csv(
            outdir / f"{symbol}_{fname}{suffix}_trades.csv",
            index=False, encoding="utf-8-sig",
        )
    else:
        print(f"[warn] {symbol}: 该区间内未产生任何交易", file=sys.stderr)

    if not args.no_plot:
        plot_equity_curve(
            window_dates, strat[eval_start:], bh[eval_start:],
            window_dates, bench_curve[eval_start:],
            symbol, label, period_label,
            outdir / f"{symbol}_{fname}{suffix}.png",
        )

    bh_metrics = compute_metrics(bh * args.capital, eval_start)
    bench_metrics = compute_metrics(bench_curve * args.capital, eval_start)

    print(f"\n===== {symbol} · {label}（{period_label}，{args.adjust} 复权）=====")
    print(pd.DataFrame({
        "策略": {
            "总收益率": _fmt_pct(metrics["total"]),
            "年化收益率": _fmt_pct(metrics["annual"]),
            "最大回撤": _fmt_pct(metrics["mdd"]),
            "胜率": f"{metrics['win_rate'] * 100:.1f}%" if not np.isnan(metrics["win_rate"]) else "—",
            "交易次数": int(metrics["trades"]),
        },
        "买入持有": {
            "总收益率": _fmt_pct(bh_metrics["total"]),
            "年化收益率": _fmt_pct(bh_metrics["annual"]),
            "最大回撤": _fmt_pct(bh_metrics["mdd"]),
            "胜率": "—",
            "交易次数": 0,
        },
        "沪深300基准": {
            "总收益率": _fmt_pct(bench_metrics["total"]),
            "年化收益率": _fmt_pct(bench_metrics["annual"]),
            "最大回撤": _fmt_pct(bench_metrics["mdd"]),
            "胜率": "—",
            "交易次数": 0,
        },
    }).to_string())  # 中文字段用 to_string 输出

    results.append({
        "symbol": symbol, "strategy": strategy.name, "params": strategy.param_string(),
        "start": args.start or str(dates.iloc[eval_start].date()),
        "end": end, "adjust": args.adjust,
        "strat_total": metrics["total"], "strat_annual": metrics["annual"],
        "strat_mdd": metrics["mdd"], "strat_win_rate": metrics["win_rate"],
        "strat_trades": int(metrics["trades"]),
        "bh_total": bh_metrics["total"], "bh_annual": bh_metrics["annual"],
        "bh_mdd": bh_metrics["mdd"],
        "bench_total": bench_metrics["total"], "bench_annual": bench_metrics["annual"],
        "bench_mdd": bench_metrics["mdd"],
        "bench_source": bench_source,
        "data_source": data_source,
        "fractional": args.fractional,
    })
    return True


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass

    args = parse_args(argv)
    try:
        strategy = build_strategy(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    end = args.end or datetime.now().strftime("%Y%m%d")
    print(f"回测区间: {args.start or '数据起点'} ~ {end} | 策略: {strategy.label()} | "
          f"复权: {args.adjust} | 初始资金: {args.capital:,.0f} | "
          f"成本: 佣金{args.commission:.2%} + 卖出印花税{args.stamp:.2%}")

    benchmark, bench_source = load_benchmark(args.start, end, args.offline, args.retries)
    print(f"[ok] 沪深300基准数据: {bench_source}，{len(benchmark)} 行")

    results: list[dict] = []
    failed: list[str] = []
    for i, raw_symbol in enumerate(args.symbols):
        if i > 0:
            time.sleep(1.0)  # 尊重上游接口的请求间隔
        try:
            match = re.search(r"\d{6}", raw_symbol.strip())
            if not match:
                raise ValueError(f"无法识别股票代码: {raw_symbol!r}")
            symbol = match.group(0)
            if not run_one(symbol, args, strategy, benchmark, bench_source, results):
                failed.append(raw_symbol)
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            print(f"[failed] {raw_symbol}: {exc}", file=sys.stderr)
            failed.append(raw_symbol)

    if results:
        suffix = f"_{args.tag}" if args.tag else ""
        summary_path = Path(args.outdir) / f"summary_{strategy.name}{suffix}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[ok] 汇总结果已保存: {summary_path}")

    print(f"\n⚠ {DISCLAIMER}")
    if failed:
        print(f"共 {len(args.symbols)} 只，失败 {len(failed)} 只: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
