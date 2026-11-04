"""5 分钟线 MA5/MA20 金叉死叉日内回测原型（第 7 步日内研究）。

用法示例：
    # 拉取研究标的最近的 5 分钟线（akshare stock_zh_a_hist_min_em，period=5）
    # 落盘 data/minute/<代码>_5min.csv，并打印/记录真实历史覆盖范围
    python scripts/intraday_backtest.py fetch

    # 运行日内回测（默认 000001 600519 600036 300750，MA5/MA20，10 万整手）
    python scripts/intraday_backtest.py backtest

    # 指定标的/参数/成本
    python scripts/intraday_backtest.py backtest 000001 600519 --short 5 --long 20 --capital 100000

    # 合成数据算法自检（不访问网络）
    python scripts/intraday_backtest.py selftest

口径（与监控台分钟线信号引擎、第 2 步日线框架保持一致，全部显式可复现）：
1. 数据：akshare stock_zh_a_hist_min_em(period=5) **不复权** 5 分钟线——监控台
   实时分钟数据同为不复权；fetch 时逐标的校验 qfq 与不复权是否一致并记录
   （窗口内无除权事件时两者完全相同）；
2. 信号：5 分钟收盘价的 MA5 上穿 MA20 → 金叉（目标仓位 1）；下穿 → 死叉
   （目标仓位 0），与 scripts/strategies.py 的 ma_cross 同一定义；
3. 成交：t 根 K 线收盘产生信号 → t+1 根 K 线开盘价成交（避免未来函数，
   与日线框架"t 收盘信号、t+1 开盘成交"一致）；
4. T+1 约束与次日卖出规则：当日买入的仓位当日不得卖出；死叉卖出信号若与
   买入处于同一交易日，顺延至下一交易日第一根 K 线开盘价卖出；顺延期间
   不响应新信号（仓位未落定）；
5. 成本：买入佣金万三（--commission，默认 0.0003）；卖出佣金万三 + 印花税
   万五（--stamp，默认 0.0005）；未建模滑点（实盘只会更差）；
6. 仓位：初始资金 10 万（--capital），整手买入（1 手 = 100 股）；--fractional
   为小数股全仓的学术口径（研究信号本身时用，与日线框架一致）；
7. 期末处理：窗口末尾仍持仓按最后收盘价强平，计入交易明细与胜率（与日线
   框架口径一致）；
8. 基准：同期买入持有（窗口首根就绪 K 线收盘买入、无成本，与日线框架 BH
   口径一致）；不引入指数基准——分钟线窗口过短，日频指数对比无意义。

输出（output/intraday/ 下）：
  - <代码>_ma5_20_equity.csv   每根 K 线的价格、MA5/MA20、净值与买入持有净值；
  - <代码>_ma5_20_trades.csv   交易明细（净价已含交易成本）；
  - <代码>_ma5_20.png          价格+均线+买卖点 / 净值对比 双面板图；
  - summary_ma5_20.csv         全部标的汇总（含数据覆盖范围与信号统计）。
报告与图表均明确标注：历史模拟回测不代表未来，不构成投资建议。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOT_SIZE = 100  # A 股买入单位：1 手 = 100 股
TRADING_DAYS = 252  # 年化收益率使用的年交易日数（沿用日线框架口径）
DEFAULT_SYMBOLS = ["000001", "600519", "600036", "300750"]
DEFAULT_SHORT, DEFAULT_LONG = 5, 20  # 与监控台信号引擎 MA5/MA20 一致

DISCLAIMER = "历史模拟回测结果不代表未来表现，不构成投资建议。"


@dataclass
class Trade:
    """一次完整的开仓→平仓往返交易，价格均为已含交易成本的净价。"""

    entry_time: pd.Timestamp
    entry_price: float  # 买入净价（含佣金）
    exit_time: pd.Timestamp
    exit_price: float  # 卖出净价（已扣佣金+印花税）
    shares: float  # 整手口径为 100 的整数倍，小数股口径为任意正数
    net_return: float  # 本次往返净收益率
    holding_bars: int  # 持有 K 线数
    holding_days: int  # 持有自然日数（卖出日 - 买入日）
    exit_reason: str


# ---------------------------------------------------------------------------
# 信号计算
# ---------------------------------------------------------------------------

def cross_signals(
    close: pd.Series, short: int = DEFAULT_SHORT, long: int = DEFAULT_LONG
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """返回 (ma5, ma20, gold, death)。

    金叉：上一根 K 线 MA5 <= MA20 且本根 MA5 > MA20（t 收盘产生信号）；
    死叉：上一根 K 线 MA5 >= MA20 且本根 MA5 < MA20。
    均线未就绪（热身期）的 K 线不产生信号——注意 pandas 中 NaN 参与比较得
    False，会误把"首个有效 K 线"当作交叉，因此用 ready 显式屏蔽热身边界。
    """
    ma5 = close.rolling(short).mean()
    ma20 = close.rolling(long).mean()
    rel = ma5 > ma20
    prev = rel.shift(1)
    ready = (ma5.notna() & ma20.notna()
             & ma5.shift(1).notna() & ma20.shift(1).notna())
    gold = prev.eq(False) & rel & ready
    death = prev.eq(True) & ~rel & ready
    return ma5, ma20, gold, death


# ---------------------------------------------------------------------------
# 回测核心
# ---------------------------------------------------------------------------

def run_intraday(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    capital: float = 100000.0,
    commission: float = 0.0003,
    stamp: float = 0.0005,
    short: int = DEFAULT_SHORT,
    long: int = DEFAULT_LONG,
    fractional: bool = False,
) -> dict:
    """执行 5 分钟线 MA5/MA20 日内回测。

    df 需含 时间/开盘/收盘 三列（fetch 落盘的格式）。返回结果字典：
    equity（每根 K 线净值数组）、trades、eval_start（评估窗口起点下标）、
    gold_count/death_count（窗口内信号数）、lot_skips、dates、close、ma5、ma20、
    bh（买入持有净值，与 equity 同起点归一）。
    """
    ts = pd.to_datetime(df["时间"]).reset_index(drop=True)
    dates = ts.dt.normalize()
    close = pd.to_numeric(df["收盘"], errors="coerce").reset_index(drop=True)
    open_ = pd.to_numeric(df["开盘"], errors="coerce").reset_index(drop=True)
    n = len(df)
    close_np = close.to_numpy(dtype=float)
    open_np = open_.to_numpy(dtype=float)

    ma5, ma20, gold, death = cross_signals(close, short, long)
    gold_np = gold.to_numpy(dtype=bool)
    death_np = death.to_numpy(dtype=bool)

    in_window = pd.Series(np.ones(n, dtype=bool), index=df.index)
    if start is not None:
        in_window &= dates >= pd.Timestamp(start)
    if end is not None:
        in_window &= dates <= pd.Timestamp(end)
    in_window_np = in_window.to_numpy(dtype=bool)
    dates_np = dates.to_numpy()

    # 信号统计口径：信号 K 线与成交 K 线都在窗口内（信号在 t-1 收盘产生、t 成交）
    sig_ok = np.zeros(n, dtype=bool)
    sig_ok[1:] = in_window_np[1:] & in_window_np[:-1]
    gold_count = int(np.count_nonzero(gold_np[:-1] & sig_ok[1:]))
    death_count = int(np.count_nonzero(death_np[:-1] & sig_ok[1:]))

    cash = float(capital)
    shares = 0
    entry_i: int | None = None
    entry_px = 0.0
    entry_date: pd.Timestamp | None = None
    pending_exit = False  # T+1：当日死叉卖出顺延至次日开盘
    trades: list[Trade] = []
    lot_skips = 0
    equity = np.empty(n)

    for i in range(n):
        # 1) 顺延卖出的执行：下一交易日第一根 K 线开盘价卖出
        if pending_exit and i > 0 and dates.iloc[i] != dates.iloc[i - 1]:
            px = open_np[i] * (1 - commission - stamp)
            trades.append(Trade(
                entry_time=ts.iloc[entry_i], entry_price=entry_px,
                exit_time=ts.iloc[i], exit_price=px, shares=shares,
                net_return=px / entry_px - 1,
                holding_bars=i - entry_i,
                holding_days=(dates.iloc[i] - entry_date).days,
                exit_reason="T+1 顺延次日开盘卖出",
            ))
            cash += shares * px
            shares = 0
            entry_i = None
            entry_date = None
            pending_exit = False

        # 2) 上一根 K 线收盘信号 → 本根 K 线开盘成交
        if i >= 1 and in_window_np[i] and in_window_np[i - 1] and not pending_exit:
            if shares == 0 and gold_np[i - 1]:
                px = open_np[i] * (1 + commission)
                if fractional:
                    shares = cash / px
                    cash -= shares * px
                    entry_i, entry_px, entry_date = i, px, dates.iloc[i]
                else:
                    lots = int((cash // px) // LOT_SIZE)
                    if lots > 0:
                        shares = lots * LOT_SIZE
                        cash -= shares * px
                        entry_i, entry_px, entry_date = i, px, dates.iloc[i]
                    else:
                        lot_skips += 1  # 信号出现但资金不足一手，无法买入
            elif shares > 0 and death_np[i - 1]:
                if dates_np[i] > entry_date:  # 非买入当日 → 可直接卖出
                    px = open_np[i] * (1 - commission - stamp)
                    trades.append(Trade(
                        entry_time=ts.iloc[entry_i], entry_price=entry_px,
                        exit_time=ts.iloc[i], exit_price=px, shares=shares,
                        net_return=px / entry_px - 1,
                        holding_bars=i - entry_i,
                        holding_days=(dates.iloc[i] - entry_date).days,
                        exit_reason="死叉卖出",
                    ))
                    cash += shares * px
                    shares = 0
                    entry_i = None
                    entry_date = None
                else:  # T+1：当日买入当日不得卖出 → 顺延至次日开盘
                    pending_exit = True
        equity[i] = cash + shares * close_np[i]

    # 期末仍持仓 → 按最后收盘价强平，计入交易明细与胜率
    if shares > 0:
        px = close_np[-1] * (1 - commission - stamp)
        trades.append(Trade(
            entry_time=ts.iloc[entry_i], entry_price=entry_px,
            exit_time=ts.iloc[-1], exit_price=px, shares=shares,
            net_return=px / entry_px - 1,
            holding_bars=n - 1 - entry_i,
            holding_days=(dates.iloc[-1] - entry_date).days,
            exit_reason="期末强平（含顺延未执行）" if pending_exit else "期末强平",
        ))
        equity[-1] = cash + shares * px

    # 评估窗口起点：窗口内首个双均线就绪的 K 线
    ready = ma5.notna().to_numpy() & ma20.notna().to_numpy()
    candidates = np.flatnonzero(ready & in_window_np)
    eval_start = int(candidates[0]) if len(candidates) else max(n - 1, 0)

    bh = close_np / close_np[eval_start]  # 同期买入持有（无成本，与日线框架口径一致）
    return {
        "equity": equity,
        "trades": trades,
        "eval_start": eval_start,
        "gold_count": gold_count,
        "death_count": death_count,
        "lot_skips": lot_skips,
        "ts": ts,
        "dates": dates,
        "close": close_np,
        "open": open_np,
        "ma5": ma5.to_numpy(),
        "ma20": ma20.to_numpy(),
        "bh": bh,
    }


def compute_metrics(result: dict) -> dict[str, float]:
    """在评估窗口上计算收益指标（与日线框架 compute_metrics 同口径）。"""
    s = result["eval_start"]
    eq = pd.Series(result["equity"][s:]).reset_index(drop=True)
    n_days = int(result["dates"].iloc[s:].nunique())
    if eq.iloc[0] <= 0 or len(eq) < 2:
        return {"total": np.nan, "annual": np.nan, "mdd": np.nan}
    total = eq.iloc[-1] / eq.iloc[0] - 1
    annual = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n_days) - 1 if n_days > 1 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": total, "annual": annual, "mdd": mdd}


def trade_stats(trades: list[Trade]) -> dict[str, float]:
    wins = sum(1 for t in trades if t.net_return > 0)
    return {
        "trades": float(len(trades)),
        "wins": float(wins),
        "win_rate": wins / len(trades) if trades else np.nan,
    }


# ---------------------------------------------------------------------------
# 数据拉取
# ---------------------------------------------------------------------------

def fetch_one(symbol: str, adjust: str, retries: int) -> pd.DataFrame:
    """拉取单只股票的 5 分钟线，失败按指数退避重试。adjust: '' 不复权 / 'qfq' 前复权。"""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 akshare，请先执行: pip install akshare pandas") from exc

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="5", adjust=adjust)
            if df is None or df.empty:
                raise ValueError("接口返回空数据（代码可能有误，或该区间内无交易）")
            return df
        except Exception as exc:  # noqa: BLE001 上游为第三方接口，异常种类多，统一重试
            last_err = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"重试 {retries} 次后仍失败: {last_err}") from last_err


def _same_ohlc(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    for col in ("开盘", "收盘", "最高", "最低"):
        if not np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                           rtol=0.0, atol=1e-9):
            return False
    return True


def cmd_fetch(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed: list[str] = []
    for raw_symbol in args.symbols:
        time.sleep(args.sleep)
        try:
            df_raw = fetch_one(raw_symbol, "", args.retries)
            time.sleep(args.sleep)
            df_qfq = fetch_one(raw_symbol, "qfq", args.retries)
            identical = _same_ohlc(df_raw, df_qfq)
            out_path = outdir / f"{raw_symbol}_5min.csv"
            df_raw.to_csv(out_path, index=False, encoding="utf-8-sig")
            if not identical:
                df_qfq.to_csv(outdir / f"{raw_symbol}_5min_qfq.csv", index=False, encoding="utf-8-sig")
            dates = pd.to_datetime(df_raw["时间"]).dt.normalize()
            rows.append({
                "代码": raw_symbol,
                "K线数": len(df_raw),
                "覆盖起点": df_raw["时间"].iloc[0],
                "覆盖终点": df_raw["时间"].iloc[-1],
                "交易日数": int(dates.nunique()),
                "qfq与不复权一致": "是" if identical else "否（已另存 qfq）",
            })
            print(f"[ok] {raw_symbol}: {len(df_raw)} 根 5 分钟 K 线, "
                  f"{df_raw['时间'].iloc[0]} ~ {df_raw['时间'].iloc[-1]}"
                  f"（{int(dates.nunique())} 个交易日）, qfq 与不复权"
                  f"{'一致' if identical else '不一致（已另存 qfq）'}, 已保存 {out_path}")
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            print(f"[failed] {raw_symbol}: {exc}", file=sys.stderr)
            failed.append(raw_symbol)

    coverage_path = outdir / "coverage_5min.csv"
    pd.DataFrame(rows).to_csv(coverage_path, index=False, encoding="utf-8-sig")
    print(f"\n覆盖范围已记录: {coverage_path}")
    if failed:
        print(f"共 {len(args.symbols)} 只，失败 {len(failed)} 只: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"全部成功：{len(args.symbols)} 只")
    return 0


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


def plot_intraday(result: dict, symbol: str, label: str, period_label: str,
                  out_png: Path) -> None:
    """双面板图：上半部分价格+均线+买卖点，下半部分策略净值 vs 买入持有。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    ts = result["ts"]
    s = result["eval_start"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(ts, result["close"], lw=0.8, color="tab:gray", label="收盘价")
    ax1.plot(ts, result["ma5"], lw=0.9, color="tab:blue", label="MA5")
    ax1.plot(ts, result["ma20"], lw=0.9, color="tab:orange", label="MA20")
    for t in result["trades"]:
        ax1.scatter([t.entry_time], [result["close"][result["ts"].searchsorted(t.entry_time)]],
                    marker="^", s=45, color="tab:green", zorder=5, label="买入" if t is result["trades"][0] else "")
        ax1.scatter([t.exit_time], [result["close"][result["ts"].searchsorted(t.exit_time)]],
                    marker="v", s=45, color="tab:red", zorder=5, label="卖出" if t is result["trades"][0] else "")
    ax1.set_title(f"{symbol} · {label}（{period_label}）")
    ax1.set_ylabel("价格（元）")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    strat = result["equity"][s:] / result["equity"][s]
    bh = result["bh"][s:]
    ax2.plot(ts[s:], strat, lw=1.2, color="tab:blue", label="策略净值")
    ax2.plot(ts[s:], bh, lw=1.0, alpha=0.7, color="tab:orange", label="买入持有（同期）")
    ax2.set_ylabel("净值（期初=1）")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def cmd_backtest(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label = f"ma{args.short}_{args.long}"
    suffix = f"_{args.tag}" if args.tag else ""

    results: list[dict] = []
    for symbol in args.symbols:
        data_path = Path(args.datadir) / f"{symbol}_5min.csv"
        if not data_path.exists():
            print(f"[failed] {symbol}: 未找到 {data_path}，请先运行 `python scripts/intraday_backtest.py fetch`",
                  file=sys.stderr)
            continue
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        result = run_intraday(
            df, start=args.start, end=args.end, capital=args.capital,
            commission=args.commission, stamp=args.stamp,
            short=args.short, long=args.long, fractional=args.fractional,
        )
        metrics = compute_metrics(result)
        stats = trade_stats(result["trades"])
        s = result["eval_start"]
        ts = result["ts"]
        dates = pd.Series(result["dates"])

        # 落盘：K 线净值 CSV
        equity_df = pd.DataFrame({
            "时间": ts,
            "开盘": result["open"],
            "收盘": result["close"],
            f"MA{args.short}": result["ma5"],
            f"MA{args.long}": result["ma20"],
            "净值": result["equity"],
            "买入持有净值": result["bh"],
        }).iloc[s:].reset_index(drop=True)
        equity_path = outdir / f"{symbol}_{label}{suffix}_equity.csv"
        equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")

        # 落盘：交易明细 CSV（0 交易时也保留表头）
        trades_cols = ["买入时间", "买入净价", "卖出时间", "卖出净价",
                       "股数", "往返收益率", "持有K线数", "持有自然日", "卖出原因"]
        trades_df = pd.DataFrame([{
            "买入时间": t.entry_time, "买入净价": round(t.entry_price, 4),
            "卖出时间": t.exit_time, "卖出净价": round(t.exit_price, 4),
            "股数": int(t.shares) if float(t.shares).is_integer() else round(t.shares, 4),
            "往返收益率": t.net_return,
            "持有K线数": t.holding_bars, "持有自然日": t.holding_days,
            "卖出原因": t.exit_reason,
        } for t in result["trades"]], columns=trades_cols)
        trades_path = outdir / f"{symbol}_{label}{suffix}_trades.csv"
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")

        # 图表
        if not args.no_plot:
            png_path = outdir / f"{symbol}_{label}{suffix}.png"
            plot_intraday(
                result, symbol, f"MA{args.short}/{args.long} 金叉死叉（5 分钟线）",
                f"{dates.iloc[0].date()} ~ {dates.iloc[-1].date()}", png_path,
            )

        bh_total = result["bh"][-1] - 1
        row = {
            "代码": symbol,
            "数据起点": df["时间"].iloc[0],
            "数据终点": df["时间"].iloc[-1],
            "K线数": len(df),
            "交易日数": int(dates.nunique()),
            "金叉信号": result["gold_count"],
            "死叉信号": result["death_count"],
            "因整手跳过": result["lot_skips"],
            "策略总收益率": metrics["total"],
            "策略年化": metrics["annual"],
            "策略最大回撤": metrics["mdd"],
            "胜率": stats["win_rate"],
            "交易次数": int(stats["trades"]),
            "盈利次数": int(stats["wins"]),
            "BH总收益率": bh_total,
        }
        results.append(row)
        print(
            f"[ok] {symbol}: 总收益 {metrics['total']:+.2%} | "
            f"BH {bh_total:+.2%} | 回撤 {metrics['mdd']:.2%} | "
            f"胜率 {stats['win_rate']:.1%} ({int(stats['trades'])} 次) | "
            f"金叉 {result['gold_count']} / 死叉 {result['death_count']}"
        )

    summary_path = outdir / f"summary_{label}{suffix}.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] 汇总结果已保存: {summary_path}")
    print(f"成本口径: 佣金 {args.commission:.2%} + 卖出印花税 {args.stamp:.2%} | "
          f"初始资金 {args.capital:,.0f} 元"
          f"{'小数股全仓（学术口径）' if args.fractional else '整手'} | 未建模滑点")
    return 0


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _make_df(bars: list[tuple[str, float]]) -> pd.DataFrame:
    """构造合成 5 分钟数据：bars 为 (时间, 收盘价)，开盘/最高/最低取收盘价。"""
    return pd.DataFrame({
        "时间": [b[0] for b in bars],
        "开盘": [b[1] for b in bars],
        "收盘": [b[1] for b in bars],
        "最高": [b[1] for b in bars],
        "最低": [b[1] for b in bars],
    })


def _ref_cross(close: np.ndarray, short: int, long: int) -> tuple[np.ndarray, np.ndarray]:
    """朴素循环实现的交叉信号（参考实现，用于对照向量化版本）。"""
    n = len(close)
    ma5 = pd.Series(close).rolling(short).mean().to_numpy()
    ma20 = pd.Series(close).rolling(long).mean().to_numpy()
    gold = np.zeros(n, dtype=bool)
    death = np.zeros(n, dtype=bool)
    for i in range(1, n):
        p5, p20, c5, c20 = ma5[i - 1], ma20[i - 1], ma5[i], ma20[i]
        if np.isnan(p5) or np.isnan(p20) or np.isnan(c5) or np.isnan(c20):
            continue
        if p5 <= p20 and c5 > c20:
            gold[i] = True
        if p5 >= p20 and c5 < c20:
            death[i] = True
    return gold, death


def _selftest_cross_detection() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        close = pd.Series(rng.uniform(9, 12, 300))
        _, _, gold, death = cross_signals(close, 5, 20)
        g_ref, d_ref = _ref_cross(close.to_numpy(), 5, 20)
        assert np.array_equal(gold.to_numpy(), g_ref), "金叉检测与参考实现不一致"
        assert np.array_equal(death.to_numpy(), d_ref), "死叉检测与参考实现不一致"


def _selftest_t_plus_1() -> None:
    # day1：10 元横盘 20 根 → 金叉买入；随后跳涨再跳水，同日出现死叉 → T+1 顺延
    bars = []
    t = pd.Timestamp("2026-10-12 09:30:00")
    for k in range(40):
        if k < 20:
            px = 10.0
        elif k < 26:
            px = 12.0 + 0.01 * (k - 20)  # 温和上行，避免 MA20 追平产生假死叉
        else:
            px = 9.0  # 跳水 → 当日死叉
        bars.append((t + pd.Timedelta(minutes=5 * k), px))
    day2_start = pd.Timestamp("2026-10-13 09:30:00")
    for k in range(20):
        bars.append((day2_start + pd.Timedelta(minutes=5 * k), 9.0))
    df = _make_df(bars)
    result = run_intraday(df, capital=100000.0, commission=0.0003, stamp=0.0005)
    trades = result["trades"]

    assert len(trades) == 1, f"应只有一笔交易，实际 {len(trades)}"
    buy = trades[0]
    # 金叉在 bar 20（11:10）收盘产生 → bar 21（11:15）开盘价 12.01 成交（含佣金）
    assert buy.entry_time == pd.Timestamp("2026-10-12 11:15:00"), f"金叉下一根 K 线开盘买入，实际 {buy.entry_time}"
    assert abs(buy.entry_price - 12.01 * 1.0003) < 1e-9, f"买入净价 {buy.entry_price}"
    # T+1：当日死叉（bar 28，11:50）不得当日卖出 → 顺延至次日首根 K 线（09:30）开盘
    assert buy.exit_time == pd.Timestamp("2026-10-13 09:30:00"), \
        f"T+1：当日死叉应顺延至次日首根 K 线开盘卖出，实际 {buy.exit_time}"
    assert buy.exit_reason == "T+1 顺延次日开盘卖出", buy.exit_reason
    assert abs(buy.exit_price - 9.0 * (1 - 0.0003 - 0.0005)) < 1e-9, f"卖出净价 {buy.exit_price}"
    assert buy.holding_days == 1, f"持有自然日 {buy.holding_days}"


def _selftest_cross_day_sell() -> None:
    # 隔日死叉：买入在 day1，死叉在 day2 → 死叉下一根 K 线直接卖出（不顺延）
    bars = []
    t = pd.Timestamp("2026-10-12 09:30:00")
    for k in range(20):
        bars.append((t + pd.Timedelta(minutes=5 * k), 10.0))
    for k in range(20, 40):
        bars.append((t + pd.Timedelta(minutes=5 * k), 12.0 + 0.01 * (k - 20)))
    day2_start = pd.Timestamp("2026-10-13 09:30:00")
    for k in range(5):
        bars.append((day2_start + pd.Timedelta(minutes=5 * k), 12.0 + 0.01 * (k + 20)))
    for k in range(5, 20):
        bars.append((day2_start + pd.Timedelta(minutes=5 * k), 9.0))
    df = _make_df(bars)
    result = run_intraday(df, capital=100000.0, commission=0.0003, stamp=0.0005)
    trades = result["trades"]
    assert len(trades) == 1, f"应只有一笔交易，实际 {len(trades)}"
    assert trades[0].exit_reason == "死叉卖出", [t.exit_reason for t in trades]
    # 死叉在 day2 出现，卖出成交于死叉信号下一根 K 线开盘（day2 内，不顺延）
    assert trades[0].exit_time.date() == pd.Timestamp("2026-10-13").date(), trades[0].exit_time


def _selftest_final_liquidation() -> None:
    # 期末仍持仓 → 按最后收盘价强平并计入交易与胜率
    bars = []
    t = pd.Timestamp("2026-10-12 09:30:00")
    for k in range(20):
        bars.append((t + pd.Timedelta(minutes=5 * k), 10.0))
    for k in range(20, 40):
        bars.append((t + pd.Timedelta(minutes=5 * k), 12.0 + 0.01 * (k - 20)))
    df = _make_df(bars)
    result = run_intraday(df, capital=100000.0, commission=0.0003, stamp=0.0005)
    trades = result["trades"]
    assert len(trades) == 1 and trades[0].exit_reason == "期末强平", [t.exit_reason for t in trades]
    stats = trade_stats(trades)
    assert stats["trades"] == 1
    expected_final = 100000.0 - trades[0].entry_price * trades[0].shares \
        + trades[0].exit_price * trades[0].shares
    assert abs(result["equity"][-1] - expected_final) < 1e-9, \
        f"期末净值 {result['equity'][-1]} 与手算 {expected_final} 不符"


def _selftest_metrics() -> None:
    # 指标口径：单调上行序列无任何交叉 → 无交易、净值恒为初始资金；
    # 评估窗口从 MA20 就绪的首根 K 线（bar 19）开始，BH 总收益 = close[-1]/close[19] - 1
    bars = []
    t = pd.Timestamp("2026-10-12 09:30:00")
    for k in range(40):
        bars.append((t + pd.Timedelta(minutes=5 * k), 10.0 + 0.05 * k))
    df = _make_df(bars)
    result = run_intraday(df, capital=100000.0, commission=0.0, stamp=0.0)
    assert result["eval_start"] == 19, f"MA20 就绪首根 K 线应为 bar 19，实际 {result['eval_start']}"
    assert not result["trades"], "单调上行不应产生交叉交易"
    assert np.all(result["equity"] == 100000.0), "无交易时净值应恒为初始资金"
    expected_bh = 11.95 / 10.95 - 1
    assert abs(result["bh"][-1] - 1 - expected_bh) < 1e-12, f"BH 总收益 {result['bh'][-1] - 1}"


def _selftest_fractional() -> None:
    # 高价股：1 手都买不起 → 整手口径 0 交易；小数股口径全仓可买（与日线框架 600519 坑一致）
    bars = []
    t = pd.Timestamp("2026-10-12 09:30:00")
    for k in range(20):
        bars.append((t + pd.Timedelta(minutes=5 * k), 200.0))
    for k in range(20, 40):
        bars.append((t + pd.Timedelta(minutes=5 * k), 210.0 + 0.01 * (k - 20)))
    df = _make_df(bars)
    whole = run_intraday(df, capital=10000.0, commission=0.0, stamp=0.0)
    assert not whole["trades"] and whole["lot_skips"] == 1, \
        f"整手口径应无交易且跳过 1 次信号，实际 {whole['trades']}/{whole['lot_skips']}"
    frac = run_intraday(df, capital=10000.0, commission=0.0, stamp=0.0, fractional=True)
    assert len(frac["trades"]) == 1 and frac["trades"][0].exit_reason == "期末强平", \
        [t.exit_reason for t in frac["trades"]]
    # 金叉在 bar 20 产生，bar 21 开盘价 210.01 全仓小数股买入
    assert abs(frac["trades"][0].shares - 10000.0 / 210.01) < 1e-9, \
        f"小数股买入股数 {frac['trades'][0].shares}"


def cmd_selftest() -> int:
    _selftest_cross_detection()
    _selftest_t_plus_1()
    _selftest_cross_day_sell()
    _selftest_final_liquidation()
    _selftest_metrics()
    _selftest_fractional()
    print("[selftest] 全部通过：交叉检测、T+1 顺延次日卖出、隔日死叉直卖、期末强平、指标口径、小数股口径")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5 分钟线 MA5/MA20 金叉死叉日内回测（T+1 约束、佣金印花税、次日卖出规则）",
        epilog=(
            "示例:\n"
            "  python scripts/intraday_backtest.py fetch\n"
            "  python scripts/intraday_backtest.py backtest 000001 600519\n"
            "  python scripts/intraday_backtest.py selftest\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="拉取 5 分钟线并记录真实覆盖范围")
    p_fetch.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS,
                         help="股票代码，默认 000001 600519 600036 300750")
    p_fetch.add_argument("--outdir", default="data/minute", help="CSV 输出目录，默认 data/minute")
    p_fetch.add_argument("--retries", type=int, default=3, help="单只失败重试次数，默认 3")
    p_fetch.add_argument("--sleep", type=float, default=1.0,
                         help="请求间隔（秒），默认 1，请勿设 0 以免对上游造成压力")

    p_bt = sub.add_parser("backtest", help="运行 MA5/MA20 金叉死叉日内回测")
    p_bt.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS,
                      help="股票代码，默认 000001 600519 600036 300750")
    p_bt.add_argument("--datadir", default="data/minute", help="分钟线 CSV 目录，默认 data/minute")
    p_bt.add_argument("--outdir", default="output/intraday", help="输出目录，默认 output/intraday")
    p_bt.add_argument("--start", default=None, help="回测开始日期 YYYYMMDD，默认数据覆盖起点")
    p_bt.add_argument("--end", default=None, help="回测结束日期 YYYYMMDD，默认数据覆盖终点")
    p_bt.add_argument("--short", type=int, default=DEFAULT_SHORT, help="短期均线窗口（K 线数），默认 5")
    p_bt.add_argument("--long", type=int, default=DEFAULT_LONG, help="长期均线窗口（K 线数），默认 20")
    p_bt.add_argument("--capital", type=float, default=100000.0, help="初始资金（元），默认 100000")
    p_bt.add_argument("--commission", type=float, default=0.0003, help="单边佣金费率，默认 0.0003（万三）")
    p_bt.add_argument("--stamp", type=float, default=0.0005, help="卖出印花税费率，默认 0.0005（万五）")
    p_bt.add_argument("--fractional", action="store_true",
                      help="允许不足一手的小数股数全仓买入（学术回测口径；默认按 A 股 1 手=100 股取整）")
    p_bt.add_argument("--tag", default=None, help="输出文件名后缀（区分不同配置）")
    p_bt.add_argument("--no-plot", action="store_true", help="不生成 PNG 图")

    sub.add_parser("selftest", help="合成数据算法自检（不访问网络）")

    args = parser.parse_args(argv)
    if args.command == "backtest":
        if args.short <= 0 or args.long <= 0:
            parser.error("--short/--long 必须为正整数")
        if args.short >= args.long:
            parser.error(f"短期窗口({args.short})必须小于长期窗口({args.long})")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    return cmd_selftest()


if __name__ == "__main__":
    sys.exit(main())
