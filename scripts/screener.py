"""横截面选股研究（量化学习项目第 6 步）。

以沪深300 + 中证500 成分股为股票池（约 800 只，全市场留作后续扩展），
用 akshare 分批限速拉取 qfq 日线（增量缓存 + 断点续跑），计算**动量**与
**波动率**因子，按月排序选出 Top-N 组合并回测，与沪深300 对比。

用法：
    # 自检（合成数据，不访问网络）
    python scripts/screener.py selftest

    # 1) 拉取成分股 + 分批限速拉取日线（增量缓存 + 断点续跑）
    python scripts/screener.py fetch

    # 查看拉取进度 / 失败清单
    python scripts/screener.py status

    # 2) 因子计算 + 月度 Top-N 回测（默认动量与低波动各一组）
    python scripts/screener.py backtest

数据坑与口径（详见 docs/optimization-log.md 第 8 章）：
  1. **停牌**：akshare 日线只返回有交易的日期，停牌日缺失 → 收盘价前向
     填充（只在该股票自身数据范围内）；调仓日若目标股票 t+1 无开盘价
     （停牌/未上市），跳过该股、份额留现金，持仓中停牌的股票冻结到下
     次调仓。
  2. **退市/成分变动（幸存者偏差）**：本步股票池取**当前**成分股快照，
     回测区间内退市或被剔除的股票不在池内——回测天然存在幸存者偏差，
     只能如实声明并留待"历史成分股"数据补齐（见日志第 8 章）。
  3. **新股**：上市日晚于因子回看窗口的股票因子为 NaN，自动排除出排序，
     满最短历史长度（--min-bars）后才进入候选。
  4. **前复权快照**：qfq 是"活数据"，未来除权除息会整体改写历史。因此
     缓存以 (fetch_start, fetch_end) 为快照键：同窗口直接复用，换窗口
     全量重拉，**禁止增量拼接**（拼接会混入两个复权基准）。
  5. **停牌对波动率的扭曲**：波动率用个股自身最近 --vol-window 个交易日
     （跳过停牌日）计算，不用并集日历前向填充序列（后者会把停牌期收益
     计为 0，系统性低估波动率）。
  6. **qfq 历史价可为 0/负数**：大额分红股的前复权早期价格可能 ≤0
     （如 300750 上市头两日 qfq 收盘 -2.09/-0.08）——价格 ≤0 的行按缺失
     处理（收盘前向填充、开盘保持 NaN 不可成交），因子计算要求价格 >0。

执行口径（与第 3/5 步引擎一致处沿用）：
  - t 日收盘后按因子排序产生目标组合 → t+1 日开盘价成交（无未来函数）；
  - 买入佣金 --commission（默认万三）；卖出佣金 + 印花税 --stamp（默认
    万五）；未建模滑点；
  - 小数股全仓分配（学术口径，第 5 步 portfolio.py 同款理由：10 万元在
    数十只股票上无法表达等权）；
  - 调仓日单边换手 = 0.5 × Σ|交易金额| / 当日净值；年化单边换手 = 历次
    调仓换手之和 / 年数；
  - 组合等权（1/N）；不可交易股票份额留现金。

调研来源（经典选股因子）见 docs/quant-factor-research.md 与本日志第 8 章，
关键文献：
  - Jegadeesh & Titman (1993), J. Finance 48(1) 65-91（动量 12-1 口径）；
  - Liu, Stambaugh & Yuan (2019), JFE（中国价值因子：E/P 优于 B/P、壳价值
    污染、剔除最小 30% 市值）；
  - Ang, Hodrick, Xing & Zhang (2006), J. Finance（低波动/特质波动率异象）；
  - Frazzini & Pedersen (2014), JFE（Betting Against Beta）。

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

# 保证以 `python scripts/screener.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    _setup_cjk_font,
    compute_metrics,
    load_benchmark,
    standardize_stock,
)

# 沪深300 + 中证500（两指数成分互不重叠）
POOL_INDEXES = ["000300", "000905"]

FETCH_START_DEFAULT = "20180101"  # 因子热身：12-1 动量需要 250+21 个交易日
FETCH_END_DEFAULT = "20251231"  # 与第 2~5 步统一窗口（项目约定）
BACKTEST_START_DEFAULT = "20200101"
BACKTEST_END_DEFAULT = "20251231"
PANEL_START = "20180101"  # 面板构建起点（恒早于回测起点，供因子热身）

FACTOR_DIR = "data/screener"
UNIVERSE_GLOB = "universe_*.csv"
FETCH_STATE = "fetch_state.csv"
FETCH_FAILURES = "fetch_failures.txt"

# akshare 中文列名（东财源），fetch_daily.py 同款落盘格式
OUT_COLS = ["日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量",
            "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]


# ---------------------------------------------------------------------------
# 股票池（成分股）
# ---------------------------------------------------------------------------

def fetch_universe(offline: bool = False, retries: int = 3) -> pd.DataFrame:
    """拉取沪深300+中证500 当前成分股并缓存（按拉取日期快照）。

    返回 DataFrame：symbol / name / index（来源指数）/ as_of。
    同一天重复运行直接复用缓存；offline 模式只用本地缓存。
    """
    import akshare as ak

    fdir = Path(FACTOR_DIR)
    fdir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    cache_path = fdir / f"universe_{today}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"symbol": str})
        if not df.empty:
            return df

    if offline:
        candidates = sorted(fdir.glob(UNIVERSE_GLOB))
        if not candidates:
            raise RuntimeError(
                f"offline 模式且无成分股缓存（{fdir / UNIVERSE_GLOB}），"
                "无法确定股票池")
        df = pd.read_csv(candidates[-1], dtype={"symbol": str})
        print(f"[warn] offline 模式，复用缓存 {candidates[-1].name}")
        return df

    frames: list[pd.DataFrame] = []
    for idx in POOL_INDEXES:
        raw = None
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw = ak.index_stock_cons_csindex(symbol=idx)
                if raw is not None and not raw.empty:
                    break
                last_err = ValueError("接口返回空成分股")
            except Exception as exc:  # noqa: BLE001 第三方接口，异常种类多
                last_err = exc
                if attempt < retries:
                    time.sleep(2 * attempt)
        if raw is None or raw.empty:
            raise RuntimeError(f"{idx} 成分股拉取失败: {last_err}")
        frames.append(pd.DataFrame({
            "symbol": raw["成分券代码"].astype(str).str.zfill(6),
            "name": raw["成分券名称"],
            "index": idx,
        }))

    df = pd.concat(frames, ignore_index=True).drop_duplicates("symbol")
    df["as_of"] = today
    df = df.sort_values("symbol").reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[ok] 股票池: 沪深300+中证500 共 {len(df)} 只（去重后），"
          f"快照 {today}，已缓存 {cache_path}")
    return df


# ---------------------------------------------------------------------------
# 日线拉取：增量缓存 + 断点续跑
# ---------------------------------------------------------------------------

def _read_state(state_path: Path) -> pd.DataFrame:
    if not state_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(state_path, dtype={"symbol": str})
    except Exception:  # noqa: BLE001 状态文件损坏视为空
        print(f"[warn] 状态文件 {state_path} 读取失败，视为空", file=sys.stderr)
        return pd.DataFrame()


def _csv_covers(csv_path: Path, start_ts: pd.Timestamp,
                end_ts: pd.Timestamp) -> bool:
    """判断已有 CSV 是否完整覆盖拉取窗口的"末端"（max >= end）。

    首端覆盖由 fetch_state 的快照键（fetch_start/fetch_end 相同即复用）保证：
    晚于窗口起点上市的新股，其 CSV 自然以上市日为起点，min <= start 恒不成立，
    不能据此重拉。此处只校验文件可读、非空且数据延伸到了窗口终点。
    注意：qfq 是活数据，增量拼接会混入两个复权基准，因此不允许"只补尾部"，
    窗口未覆盖时全量重拉。
    """
    if not csv_path.exists():
        return False
    try:
        df = standardize_stock(pd.read_csv(csv_path))
    except Exception:  # noqa: BLE001 损坏文件视为未覆盖
        return False
    if df.empty:
        return False
    return df["date"].max() >= end_ts


def _state_covers(state_row: pd.Series | None, start: str, end: str) -> bool:
    """fetch_state 里同窗口（快照键相同）且已成功/已确认空 → 复用，不重拉。"""
    if state_row is None or len(state_row) == 0:
        return False
    return (str(state_row.get("fetch_start", "")).replace("-", "") == start
            and str(state_row.get("fetch_end", "")).replace("-", "") == end
            and str(state_row.get("status", "")) in ("ok", "empty"))


def _fetch_one(symbol: str, start: str, end: str, retries: int,
               sleep_between: float) -> tuple[pd.DataFrame, str]:
    """单只股票日线：东财源优先，失败退腾讯源；返回 (df, 来源名)。

    整个 [start, end] 窗口内无任何行情（如窗口之后才上市的新股）时抛
    EmptyWindowError，调用方按"空窗口"记录而非拉取失败。
    """
    import akshare as ak

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start, end_date=end,
                                    adjust="qfq")
            if df is None or df.empty:
                raise EmptyWindowError(f"{symbol}: {start}~{end} 窗口内无行情"
                                       "（可能为窗口之后上市）")
            return df, "akshare 东财"
        except EmptyWindowError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(sleep_between + 2 * attempt)
    try:
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start,
                                   end_date=end, adjust="qfq")
        if df is None or df.empty:
            raise EmptyWindowError(f"{symbol}: {start}~{end} 窗口内无行情"
                                   "（可能为窗口之后上市）")
        df = df.rename(columns={"date": "日期", "open": "开盘", "close": "收盘",
                                "high": "最高", "low": "最低",
                                "amount": "成交量"})
        return df, "akshare 腾讯（东财失败后回退）"
    except EmptyWindowError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"东财重试 {retries} 次失败（{last_err}），腾讯回退也失败: {exc}"
        ) from exc


class EmptyWindowError(ValueError):
    """拉取窗口内无任何行情（新股在窗口之后上市等），非传输失败。"""


def run_fetch(universe: pd.DataFrame, start: str, end: str, datadir: str,
              sleep_between: float, retries: int, max_stocks: int | None,
              offline: bool) -> int:
    """分批限速拉取股票池日线，增量缓存 + 断点续跑。

    缓存键 = (start, end) 快照：已覆盖窗口的股票直接跳过（断点续跑），
    失败清单记录在 data/screener/fetch_failures.txt，下次运行自动重试。
    """
    if offline:
        raise RuntimeError("fetch 需要联网，不能与 --offline 同时使用")

    outdir = Path(datadir)
    outdir.mkdir(parents=True, exist_ok=True)
    fdir = Path(FACTOR_DIR)
    fdir.mkdir(parents=True, exist_ok=True)
    state_path = fdir / FETCH_STATE
    failures_path = fdir / FETCH_FAILURES
    state = _read_state(state_path)
    if state.empty:
        state_idx = None
    else:
        state_idx = {row["symbol"]: row for _, row in state.iterrows()}

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    symbols = universe["symbol"].tolist()
    if max_stocks:
        symbols = symbols[:max_stocks]

    failed: list[tuple[str, str]] = []
    ok = cached = 0
    t_start = time.time()
    rows_out: list[dict] = []
    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(sleep_between)
        row = universe[universe["symbol"] == symbol].iloc[0]
        name, idx_src = row["name"], row["index"]
        csv_path = outdir / f"{symbol}_qfq.csv"
        status, first, last, source, n_rows = "ok", None, None, "本地缓存", None
        try:
            prev = state_idx.get(symbol) if state_idx else None
            if _state_covers(prev, start, end):
                if csv_path.exists() and _csv_covers(csv_path, start_ts,
                                                     end_ts):
                    df = standardize_stock(pd.read_csv(csv_path))
                    cached += 1
                    first, last, n_rows = (df["date"].iloc[0],
                                           df["date"].iloc[-1], len(df))
                elif str(prev.get("status", "")) == "empty":
                    # 已确认窗口内无行情：跳过，不重复请求
                    status, source, n_rows = "empty", "窗口内无行情", 0
                else:
                    df, source = _fetch_one(symbol, start, end, retries,
                                            sleep_between)
                    df = df.reindex(columns=OUT_COLS)  # 缺列补 NaN
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    ok += 1
                    first = pd.Timestamp(df["日期"].iloc[0])
                    last = pd.Timestamp(df["日期"].iloc[-1])
                    n_rows = len(df)
            else:
                df, source = _fetch_one(symbol, start, end, retries,
                                        sleep_between)
                df = df.reindex(columns=OUT_COLS)  # 缺列补 NaN
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                ok += 1
                first = pd.Timestamp(df["日期"].iloc[0])
                last = pd.Timestamp(df["日期"].iloc[-1])
                n_rows = len(df)
        except EmptyWindowError as exc:
            # 窗口内无行情（窗口之后上市的新股等）：记录为 empty，不算失败
            status, source, n_rows = "empty", "窗口内无行情", 0
            print(f"[empty] {symbol} {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            status, source = "failed", "—"
            failed.append((symbol, str(exc)[:120]))
            print(f"[failed] {symbol} {name}: {exc}", file=sys.stderr)
        rows_out.append({
            "symbol": symbol, "name": name, "index": idx_src,
            "rows": n_rows,
            "first_date": None if first is None else first.strftime("%Y-%m-%d"),
            "last_date": None if last is None else last.strftime("%Y-%m-%d"),
            "source": source,
            "fetch_start": start, "fetch_end": end,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
        })
        if (i + 1) % 25 == 0 or i == len(symbols) - 1:
            el = time.time() - t_start
            print(f"[进度] {i + 1}/{len(symbols)} 已处理，"
                  f"新拉 {ok}、缓存 {cached}、失败 {len(failed)}，"
                  f"用时 {el:.0f}s")

    pd.DataFrame(rows_out).to_csv(state_path, index=False, encoding="utf-8-sig")
    if failed:
        with failures_path.open("w", encoding="utf-8") as fh:
            for symbol, err in failed:
                fh.write(f"{symbol}\t{err}\n")
        print(f"[warn] 共 {len(symbols)} 只，失败 {len(failed)} 只，清单已写 "
              f"{failures_path}；重新运行本命令会自动重试失败项",
              file=sys.stderr)
        return 1
    print(f"[ok] 全部 {len(symbols)} 只完成：新拉 {ok}、缓存 {cached}")
    failures_path.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------------------
# 面板与因子
# ---------------------------------------------------------------------------

def build_panel(symbols: list[str], datadir: str, adjust: str,
                start: str, end: str) -> tuple[pd.DatetimeIndex, pd.DataFrame,
                                               pd.DataFrame]:
    """把单票 CSV 拼成面板。

    返回 (dates, close, open)。dates 为所有股票交易日的并集（限制在
    [start, end]）；close/open 为 dates × symbols 矩阵：
    - close：上市日前为 NaN（新股），停牌日按该股票自身数据范围前向
      填充（停牌），数据自然截止后的日期为 NaN（退市——本步股票池中
      不会出现，但代码路径保留该语义以防后续扩展到历史成分股）；
    - open：**不填充**，停牌/未上市日保持 NaN，用于判断 t+1 是否可交易
      （前向填充的旧开盘价不能当作可成交价）。
    """
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for symbol in symbols:
        csv_path = Path(datadir) / f"{symbol}_{adjust}.csv"
        if not csv_path.exists():
            continue
        try:
            df = standardize_stock(pd.read_csv(csv_path))
        except Exception:  # noqa: BLE001 单票损坏跳过
            print(f"[warn] {symbol}: CSV 读取失败，跳过", file=sys.stderr)
            continue
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if df.empty:
            continue
        # 数据坑：qfq 前复权历史价可为 0/负数（大额分红股的早期复权基准
        # 问题，如 300750 上市头两日 qfq 收盘 -2.09/-0.08）。价格 <=0 的
        # 行按缺失处理：收盘价前向填充，开盘价保持 NaN（不可成交）。
        for col in ("open", "close"):
            df.loc[df[col] <= 0, col] = np.nan
        df = df.dropna(subset=["close"])
        s_close = df.set_index("date")["close"]
        s_open = df.set_index("date")["open"]
        closes[symbol] = s_close[~s_close.index.duplicated(keep="last")]
        opens[symbol] = s_open[~s_open.index.duplicated(keep="last")]

    union = pd.DatetimeIndex(sorted(
        set().union(*(s.index for s in closes.values()))))
    close_df = pd.DataFrame(closes, index=union).ffill()
    open_df = pd.DataFrame(opens, index=union)
    return union, close_df, open_df


def factor_momentum(close: np.ndarray, t: int, lookback: int, skip: int,
                    min_bars: int) -> np.ndarray:
    """动量因子（Jegadeesh & Titman 1993 的 12-1 口径）：

        mom_j = close[t - skip] / close[t - lookback] - 1

    close 为 n_dates × n_stocks（停牌已前向填充，上市前为 NaN）。
    t 为并集日历行下标；只使用 <= t 的信息（无未来函数）。
    上市日至今有效行数不足 min_bars 的股票返回 NaN（新股排除）。
    """
    n_stocks = close.shape[1]
    out = np.full(n_stocks, np.nan)
    a, b = t - skip, t - lookback
    if a < 0 or b < 0:
        return out
    pa, pb = close[a], close[b]
    valid = ~np.isnan(pa) & ~np.isnan(pb) & (pa > 0) & (pb > 0)
    if valid.any():
        hist = np.sum(~np.isnan(close[: t + 1]), axis=0)
        valid &= hist >= min_bars
        out[valid] = pa[valid] / pb[valid] - 1.0
    return out


def factor_volatility(close: np.ndarray, t: int, window: int,
                      min_bars: int) -> np.ndarray:
    """波动率因子：个股**自身**最近 window 个交易日（跳过停牌日）的
    日收益率标准差（排序只需相对大小，此处返回日频 std）。

    不用并集日历前向填充序列计算——停牌期收益会被前向填充计成 0，
    系统性低估停牌多的股票的波动率（数据坑之一）。
    """
    n_stocks = close.shape[1]
    out = np.full(n_stocks, np.nan)
    for j in range(n_stocks):
        col = close[: t + 1, j]
        valid = col[~np.isnan(col)]
        valid = valid[valid > 0]
        if valid.size < min_bars:
            continue
        v = valid[-window:]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = v[1:] / v[:-1] - 1.0
        if len(rets) >= 2:
            out[j] = float(rets.std(ddof=1))
    return out


def select_top(values: np.ndarray, n: int, direction: int) -> list[int]:
    """按因子值选 Top-N。direction=+1 选最大（动量），-1 选最小（低波动）。
    有效值不足 N 只时选全部有效。"""
    valid_idx = np.flatnonzero(~np.isnan(values))
    if valid_idx.size == 0:
        return []
    # direction=+1：按 -values 升序 = 按 values 降序（最大在前）；
    # direction=-1：按 +values 升序（最小在前）
    order = valid_idx[np.argsort(-direction * values[valid_idx])]
    return order[:n].tolist()


def rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个交易日（并集日历内）作为调仓日。"""
    s = dates.to_series()
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).max().sort_index())


# ---------------------------------------------------------------------------
# 月度 Top-N 回测
# ---------------------------------------------------------------------------

def run_backtest(
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
    benchmark: pd.DataFrame,
) -> dict:
    """月度因子排序 → Top-N 等权组合回测。

    每类因子（momentum=选最大、volatility=选最小）各回测一组：
    - 调仓日 t（月末）收盘后计算因子 → 选出 Top-N；
    - t+1 开盘价成交；目标股票无开盘价（停牌/未上市）则跳过，份额留
      现金；持仓中停牌的股票本次不卖（冻结）；
    - 每日净值 = 现金 + Σ 持仓 × 收盘价（停牌前向填充价）。
    返回 dict：results{factor_name: {equity, turnover_annual, trade_days,
    holdings, n_holdings_mean}}，以及 ew_pool（等权池基准，月度再平衡）
    与 bench（沪深300）曲线，均截取回测评估段。
    """
    close_np = close.to_numpy(dtype=float)
    open_np = open_.to_numpy(dtype=float)
    n, m = close_np.shape
    start = pd.Timestamp(args.start)
    eval_start = int(np.searchsorted(dates, start))
    dates_ev = dates[eval_start:]
    if len(dates_ev) == 0:
        raise RuntimeError(f"回测起点 {args.start} 之后无交易日")
    n_ev = len(dates_ev)
    years = max((dates_ev[-1] - dates_ev[0]).days / 365.25, 1e-9)

    # 沪深300基准：对齐并集日历（含热身段），评估段归一
    bench_s = benchmark.set_index("date")["close"].sort_index()
    bench_aligned = bench_s.reindex(dates).ffill().bfill().to_numpy(float)
    bench_curve = bench_aligned[eval_start:] / bench_aligned[eval_start]

    # 调仓日 = 评估段内每月最后一个交易日（全矩阵行下标）
    reb_dates = rebalance_dates(dates_ev)
    reb_idx = [int(np.searchsorted(dates, d)) for d in reb_dates]

    def _run_one(factor_name: str) -> dict:
        capital = float(args.capital)
        cash = capital
        shares = np.zeros(m)
        equity = np.full(n_ev, np.nan)
        equity[0] = capital
        turnover = np.zeros(n_ev)
        trade_days = 0
        holdings: list[dict] = []

        for t in reb_idx:
            k = t + 1  # 执行日在并集日历中的下一交易日
            if k >= n:
                break  # 月末是最后一天，无 t+1 可执行
            k_ev = k - eval_start
            if factor_name == "momentum":
                vals = factor_momentum(close_np, t, args.mom_lookback,
                                       args.mom_skip, args.min_bars)
                selected = select_top(vals, args.top_n, direction=+1)
            else:
                vals = factor_volatility(close_np, t, args.vol_window,
                                         args.min_bars)
                selected = select_top(vals, args.top_n, direction=-1)
            t_open = open_np[k]  # 原始开盘价：NaN=停牌/未上市，不可交易
            tradable = [j for j in selected
                        if not np.isnan(t_open[j]) and t_open[j] > 0]
            holdings.append({
                "date": dates_ev[k_ev - 1].strftime("%Y-%m-%d"),
                "n_holdings": len(tradable),
                "symbols": " ".join(symbols[j] for j in sorted(tradable)),
            })
            if not tradable:
                continue

            # 交易前净值：可交易股票按开盘价、停牌持仓按上一收盘价估值
            eff_open = np.where(np.isnan(t_open), close_np[t], t_open)
            nav_before = cash + float(np.nansum(shares * eff_open))
            target_w = 1.0 / len(tradable)
            trades_value = 0.0
            trade_days += 1

            # 卖出：当前持仓中不在新组合里、且当日可交易的
            for j in np.flatnonzero(shares > 0):
                if j not in tradable and not np.isnan(t_open[j]):
                    sell_value = float(shares[j] * t_open[j])
                    fee = sell_value * (args.commission + args.stamp)
                    cash += sell_value - fee
                    trades_value += sell_value
                    shares[j] = 0.0

            # 买入/调平到等权
            for j in tradable:
                target_value = nav_before * target_w
                delta = target_value - shares[j] * t_open[j]
                if delta > 0:
                    fee = delta * args.commission
                    if delta + fee > cash:
                        delta = max(cash / (1.0 + args.commission), 0.0)
                        fee = delta * args.commission
                    if delta > 0:
                        shares[j] += delta / t_open[j]
                        cash -= delta + fee
                        trades_value += delta
                elif delta < 0:
                    sell_value = -delta
                    fee = sell_value * (args.commission + args.stamp)
                    shares[j] -= sell_value / t_open[j]
                    cash += sell_value - fee
                    trades_value += sell_value

            turnover[k_ev] = 0.5 * trades_value / nav_before

        for k_ev in range(1, n_ev):
            k = eval_start + k_ev
            equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
        equity = pd.Series(equity).ffill().to_numpy()
        return {
            "equity": equity,
            "turnover_annual": float(np.sum(turnover) / years),
            "trade_days": trade_days,
            "holdings": pd.DataFrame(holdings),
            "n_holdings_mean": float(np.mean(
                [h["n_holdings"] for h in holdings])) if holdings else np.nan,
        }

    results: dict[str, dict] = {}
    for factor_name in args.factors:
        results[factor_name] = _run_one(factor_name)

    # 等权池基准：当日有数据且 t+1 可交易的全部股票，月度再平衡，同成本
    capital = float(args.capital)
    cash = capital
    shares = np.zeros(m)
    ew_equity = np.full(n_ev, np.nan)
    ew_equity[0] = capital
    for t in reb_idx:
        k = t + 1
        if k >= n:
            break
        k_ev = k - eval_start
        t_open = open_np[k]
        t_close = close_np[t]
        tradable = [j for j in range(m)
                    if not np.isnan(t_open[j]) and not np.isnan(t_close[j])
                    and t_open[j] > 0]
        if not tradable:
            continue
        eff_open = np.where(np.isnan(t_open), close_np[t], t_open)
        nav_before = cash + float(np.nansum(shares * eff_open))
        target_w = 1.0 / len(tradable)
        for j in np.flatnonzero(shares > 0):
            if j not in tradable and not np.isnan(t_open[j]):
                sell_value = float(shares[j] * t_open[j])
                cash += sell_value - sell_value * (args.commission + args.stamp)
                shares[j] = 0.0
        for j in tradable:
            delta = nav_before * target_w - shares[j] * t_open[j]
            if delta > 0:
                fee = delta * args.commission
                if delta + fee > cash:
                    delta = max(cash / (1.0 + args.commission), 0.0)
                    fee = delta * args.commission
                if delta > 0:
                    shares[j] += delta / t_open[j]
                    cash -= delta + fee
            elif delta < 0:
                sell_value = -delta
                cash += sell_value - sell_value * (args.commission + args.stamp)
                shares[j] -= sell_value / t_open[j]
    for k_ev in range(1, n_ev):
        k = eval_start + k_ev
        ew_equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
    ew_equity = pd.Series(ew_equity).ffill().to_numpy()

    return {
        "dates": dates_ev,
        "results": results,
        "ew_pool": ew_equity,
        "bench": bench_curve,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

FACTOR_LABELS = {"momentum": "动量 Top-N", "volatility": "低波动 Top-N"}


def write_outputs(out: dict, args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = out["dates"]

    curves = pd.DataFrame({"date": dates})
    metrics_rows: list[dict] = []
    for factor_name, res in out["results"].items():
        label = FACTOR_LABELS[factor_name]
        m = compute_metrics(res["equity"], 0)
        curves[f"{factor_name}_nav"] = res["equity"] / res["equity"][0]
        metrics_rows.append({
            "series": label, "type": "strategy",
            "total": m["total"], "annual": m["annual"], "mdd": m["mdd"],
            "turnover_annual": res["turnover_annual"],
            "trade_days": res["trade_days"],
            "n_holdings_mean": res["n_holdings_mean"],
        })
        if not res["holdings"].empty:
            res["holdings"].to_csv(
                outdir / f"holdings_{factor_name}.csv",
                index=False, encoding="utf-8-sig")

    m_ew = compute_metrics(out["ew_pool"], 0)
    m_bench = compute_metrics(out["bench"], 0)
    curves["ew_pool_nav"] = out["ew_pool"] / out["ew_pool"][0]
    curves["csi300_nav"] = out["bench"]
    metrics_rows += [
        {"series": "等权股票池基准", "type": "benchmark",
         "total": m_ew["total"], "annual": m_ew["annual"], "mdd": m_ew["mdd"],
         "turnover_annual": np.nan, "trade_days": np.nan,
         "n_holdings_mean": np.nan},
        {"series": "沪深300基准", "type": "benchmark",
         "total": m_bench["total"], "annual": m_bench["annual"],
         "mdd": m_bench["mdd"], "turnover_annual": np.nan,
         "trade_days": np.nan, "n_holdings_mean": np.nan},
    ]
    summary = pd.DataFrame(metrics_rows)
    curves.to_csv(outdir / "equity_curves.csv", index=False,
                  encoding="utf-8-sig")
    summary.to_csv(outdir / "summary_screener.csv", index=False,
                   encoding="utf-8-sig")

    table: dict[str, dict[str, str]] = {}
    for row in metrics_rows:
        table[row["series"]] = {
            "总收益率": _fmt_pct(row["total"]),
            "年化收益率": _fmt_pct(row["annual"]),
            "最大回撤": _fmt_pct(row["mdd"]),
            "年化单边换手": _fmt_pct(row["turnover_annual"])
            if isinstance(row["turnover_annual"], float) and
            not np.isnan(row["turnover_annual"]) else "—",
            "平均持仓数": f"{row['n_holdings_mean']:.1f}"
            if isinstance(row["n_holdings_mean"], float) and
            not np.isnan(row["n_holdings_mean"]) else "—",
        }
    print(f"\n===== 月度 Top-N 选股回测 · {args.start} ~ {args.end}，"
          f"Top-N={args.top_n}，qfq，初始 {args.capital:,.0f} 元 =====")
    print(pd.DataFrame(table).to_string())

    if not args.no_plot:
        plot_compare(curves, dates, args)


def plot_compare(curves: pd.DataFrame, dates: pd.DatetimeIndex,
                 args: argparse.Namespace) -> None:
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    colors = {"momentum": "tab:blue", "volatility": "tab:red"}
    for factor_name in args.factors:
        nav = curves[f"{factor_name}_nav"]
        ax1.plot(dates, nav, lw=1.4, color=colors[factor_name],
                 label=FACTOR_LABELS[factor_name])
        dd = nav / nav.cummax() - 1
        ax2.plot(dates, dd, lw=0.8, color=colors[factor_name], alpha=0.7)
    ax1.plot(dates, curves["ew_pool_nav"], lw=1.1, color="tab:orange",
             alpha=0.75, label="等权股票池基准")
    ax1.plot(dates, curves["csi300_nav"], lw=1.1, color="tab:green",
             alpha=0.75, label="沪深300基准")
    ax1.set_title(f"横截面选股 · 月度 Top-{args.top_n}（{args.start} ~ "
                  f"{args.end}，qfq）")
    ax1.set_ylabel("净值（期初=1）")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png = Path(args.outdir) / "screener_compare.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 对比图已保存: {out_png}")


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


# ---------------------------------------------------------------------------
# 自检（合成数据，不访问网络）
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(7)
    days = 400
    base = np.arange(days)

    def synth(growth: float, noise: float) -> np.ndarray:
        return 100.0 * np.cumprod(
            1.0 + growth + noise * rng.standard_normal(days))

    n_stocks = 6
    close = np.full((days, n_stocks), np.nan)
    # A0/A1: 稳定上行（高动量、低波动）
    close[:, 0] = synth(0.002, 0.0005)
    close[:, 1] = synth(0.0015, 0.0005)
    # B0/B1: 宽幅震荡（高波动、零动量）
    close[:, 2] = 100 + 30 * np.sin(base / 5)
    close[:, 3] = 100 + 30 * np.sin(base / 7 + 1)
    # C0: 稳定下行（负动量、低波动）
    close[:, 4] = synth(-0.001, 0.0005)
    # D0: 上市晚（150 天后才有数据）→ 早期应被 min_bars 排除
    close[150:, 5] = 100 + 0.02 * base[150:]
    # 停牌：D0 在 300~320 天停牌 → 收盘价前向填充
    close[300:320, 5] = np.nan
    close_df = pd.DataFrame(close).ffill().to_numpy()

    # 1) 动量因子：t=250 时 A0/A1 显著为正、C0 为负、D0（历史不足）排除
    mom = factor_momentum(close_df, 250, lookback=250, skip=21, min_bars=250)
    assert not np.isnan(mom[0]) and not np.isnan(mom[1])
    assert mom[0] > 0.5 and mom[1] > 0.3, f"动量因子未识别上涨股: {mom}"
    assert mom[4] < -0.1, f"动量因子未识别下跌股: {mom[4]}"
    assert np.isnan(mom[5]), "上市不满 min_bars 的股票不应参与排序"
    top2 = select_top(mom, 2, direction=+1)
    assert sorted(top2) == [0, 1], f"动量 Top-2 应为 [0,1]，实际 {top2}"

    # 2) 波动率因子：震荡股波动率显著更高；D0 在 t=200（仅 51 根）排除
    vol = factor_volatility(close_df, 250, window=60, min_bars=60)
    assert vol[2] > 0.03 and vol[3] > 0.03, f"震荡股波动率应显著更高: {vol}"
    assert vol[0] < 0.01 and vol[4] < 0.01, f"趋势股波动率应显著更低: {vol}"
    vol_early = factor_volatility(close_df, 200, window=60, min_bars=60)
    assert np.isnan(vol_early[5]), "上市不满 min_bars 的股票不应参与波动率排序"
    low2 = select_top(vol, 2, direction=-1)
    assert not set(low2) & {2, 3}, f"低波动不应选中震荡股: {low2}"

    # 3) 无未来函数：factor_momentum 只用 <= t 的行——把 t+1 起的数据改掉，
    #    因子值不得变化
    tampered = close_df.copy()
    tampered[251:] = 999.0
    mom_t = factor_momentum(tampered, 250, lookback=250, skip=21, min_bars=250)
    assert np.allclose(mom_t[~np.isnan(mom_t)], mom[~np.isnan(mom)],
                       equal_nan=True), "因子计算使用了 t 之后的信息"

    # 4) 停牌前向填充：D0 在 300~320 填充后值连续
    assert not np.isnan(close_df[310, 5]), "停牌期收盘价应前向填充"

    # 5) 月度调仓日：构造已知日历
    cal = pd.DatetimeIndex(["2020-01-02", "2020-01-30", "2020-01-31",
                            "2020-02-28", "2020-03-02", "2020-03-31"])
    reb = rebalance_dates(cal)
    assert list(reb) == [pd.Timestamp("2020-01-31"),
                         pd.Timestamp("2020-02-28"),
                         pd.Timestamp("2020-03-31")], f"月末识别错误: {list(reb)}"

    # 6) 迷你回测手工验算（零成本与含成本各一组）
    #    日历：2020-01-31(t=0) / 02-03 / 02-28(t=2) / 03-02(t=3,执行日)
    d2 = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-28",
                           "2020-03-02"])
    c2 = pd.DataFrame({"s0": [10.0, 11.0, 12.0, 13.0],
                       "s1": [20.0, 10.0, 30.0, 15.0]}, index=d2)
    o2 = pd.DataFrame({"s0": [10.0, 11.0, 12.0, 12.5],
                       "s1": [20.0, 10.0, 30.0, 21.0]}, index=d2)
    bench = pd.DataFrame({"date": d2, "close": [1.0, 1.0, 1.0, 1.0]})

    # 6a) 动量 Top-1：t=2 时 mom(s0)=12/10-1=+0.2，mom(s1)=30/20-1=+0.5
    #     → 选 s1；t+1=03-02 开盘 21 元全仓买入，收盘 15 元：
    #     净值 = 10000/21*15 = 7142.857（零成本）
    mini_mom = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0, stamp=0.0, factors=["momentum"],
        mom_lookback=2, mom_skip=0, vol_window=60, min_bars=2, top_n=1)
    out_mom = run_backtest(d2, c2, o2, ["s0", "s1"], mini_mom, bench)
    expected_mom = 10000.0 / 21.0 * 15.0
    got_mom = out_mom["results"]["momentum"]["equity"][-1]
    assert abs(got_mom - expected_mom) < 0.01, \
        f"动量迷你回测 {got_mom} 与手算 {expected_mom} 不符"
    assert out_mom["results"]["momentum"]["holdings"]["symbols"].iloc[1] \
        == "s1", "动量 Top-1 应选 s1"

    # 6b) 低波动 Top-1：s0 波动远小于 s1 → 选 s0；开盘 12.5 全仓，
    #     收盘 13 → 净值 = 10000/12.5*13 = 10400（零成本）
    mini_vol = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0, stamp=0.0, factors=["volatility"],
        mom_lookback=250, mom_skip=21, vol_window=60, min_bars=2, top_n=1)
    out_vol = run_backtest(d2, c2, o2, ["s0", "s1"], mini_vol, bench)
    expected_vol = 10000.0 / 12.5 * 13.0
    got_vol = out_vol["results"]["volatility"]["equity"][-1]
    assert abs(got_vol - expected_vol) < 0.01, \
        f"低波动迷你回测 {got_vol} 与手算 {expected_vol} 不符"
    assert out_vol["results"]["volatility"]["holdings"]["symbols"].iloc[1] \
        == "s0", "低波动 Top-1 应选 s0"

    # 6c) 成本口径：佣金万三（单边买入），低波动选 s0：
    #     买入额受现金约束 = 10000/1.0003，股数 = 该值/12.5，
    #     期末净值 = 9997.0009/12.5*13 = 10396.88
    mini_cost = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0003, stamp=0.0005, factors=["volatility"],
        mom_lookback=250, mom_skip=21, vol_window=60, min_bars=2, top_n=1)
    out_cost = run_backtest(d2, c2, o2, ["s0", "s1"], mini_cost, bench)
    expected_cost = 10000.0 / 1.0003 / 12.5 * 13.0
    got_cost = out_cost["results"]["volatility"]["equity"][-1]
    assert abs(got_cost - expected_cost) < 0.01, \
        f"成本口径 {got_cost} 与手算 {expected_cost} 不符"

    # 6d) 停牌执行日：s0 在 03-02 停牌（open NaN）→ 不可交易，应跳过
    o2_susp = o2.copy()
    o2_susp.loc["2020-03-02", "s0"] = np.nan
    out_susp = run_backtest(d2, c2, o2_susp, ["s0", "s1"], mini_vol, bench)
    assert out_susp["results"]["volatility"]["holdings"]["n_holdings"].iloc[1] \
        == 0, "执行日停牌的股票不应买入"
    assert out_susp["results"]["volatility"]["equity"][-1] == 10000.0, \
        "停牌导致未建仓时净值应保持为现金"

    print("[ok] self_test 通过：因子计算、无未来函数、停牌填充、月末识别、"
          "迷你回测、成本口径与停牌执行日断言全部成立")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="横截面选股研究：沪深300+中证500 池，动量/波动率因子"
                    "月度 Top-N 回测（量化学习项目第 6 步）",
        epilog=(
            "示例:\n"
            "  python scripts/screener.py fetch\n"
            "  python scripts/screener.py backtest\n"
            "  python scripts/screener.py backtest --factor momentum "
            "--top-n 30\n"
            "  python scripts/screener.py selftest"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch",
                             help="拉取成分股与日线（增量缓存+断点续跑）")
    p_fetch.add_argument("--fetch-start", default=FETCH_START_DEFAULT,
                         help=f"日线拉取起点 YYYYMMDD，默认 "
                              f"{FETCH_START_DEFAULT}（因子热身）")
    p_fetch.add_argument("--fetch-end", default=FETCH_END_DEFAULT,
                         help=f"日线拉取终点 YYYYMMDD，默认 "
                              f"{FETCH_END_DEFAULT}（项目统一窗口）")
    p_fetch.add_argument("--sleep", type=float, default=0.3,
                         help="两只股票之间的请求间隔（秒），默认 0.3，"
                              "勿设 0 以免上游限流")
    p_fetch.add_argument("--retries", type=int, default=3,
                         help="单只股票失败重试次数，默认 3")
    p_fetch.add_argument("--max-stocks", type=int, default=None,
                         help="只处理前 N 只（调试/上游限流时分批用，"
                              "续跑自动跳过已完成）")

    sub.add_parser("status", help="查看拉取进度与失败清单")

    p_bt = sub.add_parser("backtest", help="因子计算 + 月度 Top-N 回测")
    p_bt.add_argument("--start", default=BACKTEST_START_DEFAULT,
                      help=f"回测开始日期 YYYYMMDD，默认 "
                           f"{BACKTEST_START_DEFAULT}")
    p_bt.add_argument("--end", default=BACKTEST_END_DEFAULT,
                      help=f"回测结束日期 YYYYMMDD，默认 {BACKTEST_END_DEFAULT}")
    p_bt.add_argument("--factor", default="momentum,volatility",
                      help="因子，逗号分隔: momentum（动量，选最大）、"
                           "volatility（波动率，选最小=低波动）；"
                           "默认两者都跑")
    p_bt.add_argument("--top-n", type=int, default=50,
                      help="每月选入组合的股票数 Top-N，默认 50")
    p_bt.add_argument("--mom-lookback", type=int, default=250,
                      help="动量回看窗口（交易日），默认 250（约 1 年）")
    p_bt.add_argument("--mom-skip", type=int, default=21,
                      help="动量跳过的最近交易日数，默认 21（J-T 12-1 口径）")
    p_bt.add_argument("--vol-window", type=int, default=60,
                      help="波动率估计窗口（个股自身交易日），默认 60")
    p_bt.add_argument("--min-bars", type=int, default=None,
                      help="参与排序所需的最短历史交易日数，默认取 "
                           "max(mom-lookback, vol-window)（排除新股）")
    p_bt.add_argument("--adjust", choices=["qfq", "hfq", "none"], default="qfq",
                      help="复权方式: qfq 前复权(默认) / hfq / none")
    p_bt.add_argument("--datadir", default="data/daily",
                      help="日线 CSV 目录，默认 data/daily")
    p_bt.add_argument("--outdir", default="output/screener",
                      help="输出目录，默认 output/screener")
    p_bt.add_argument("--capital", type=float, default=100000.0,
                      help="初始资金（元），默认 100000")
    p_bt.add_argument("--commission", type=float, default=0.0003,
                      help="单边佣金费率，默认 0.0003（万三）")
    p_bt.add_argument("--stamp", type=float, default=0.0005,
                      help="卖出印花税费率，默认 0.0005（万五）")
    p_bt.add_argument("--retries", type=int, default=3,
                      help="基准数据拉取失败重试次数，默认 3")
    p_bt.add_argument("--offline", action="store_true",
                      help="只用本地 CSV，不访问网络")
    p_bt.add_argument("--no-plot", action="store_true", help="不生成 PNG 对比图")

    sub.add_parser("selftest", help="只运行合成数据上的算法自检（不拉数据）")

    args = parser.parse_args(argv)

    if args.command == "backtest":
        factors = [f.strip() for f in args.factor.split(",") if f.strip()]
        bad = [f for f in factors if f not in FACTOR_LABELS]
        if bad:
            parser.error(f"未知因子 {bad}，可选: momentum, volatility")
        args.factors = factors
        if args.min_bars is None:
            args.min_bars = max(args.mom_lookback, args.vol_window)
        if args.top_n <= 0:
            parser.error("--top-n 必须为正整数")
    if args.command == "fetch" and args.sleep <= 0:
        parser.error("--sleep 必须为正（勿对上游造成压力）")
    return args


def cmd_status() -> int:
    fdir = Path(FACTOR_DIR)
    state = _read_state(fdir / FETCH_STATE)
    failures = fdir / FETCH_FAILURES
    if state.empty:
        print("尚无拉取状态（请先运行: python scripts/screener.py fetch）")
        return 0
    print(f"拉取状态（{len(state)} 只）:")
    print(state["status"].value_counts().to_string())
    if "rows" in state.columns and state["rows"].notna().any():
        r = state["rows"].dropna()
        print(f"行数分布: min={r.min():.0f} median={r.median():.0f} "
              f"max={r.max():.0f}")
    if failures.exists():
        print(f"\n失败清单（{failures}）:")
        print(failures.read_text(encoding="utf-8")[:2000])
    return 0


def _load_benchmark_cached(args: argparse.Namespace, datadir: str
                           ) -> tuple[pd.DataFrame, str]:
    """沪深300基准：优先读 data/daily/000300_benchmark.csv（date/close），
    覆盖不足时经 backtest.load_benchmark 拉取并合并落盘。"""
    cache_path = Path(datadir) / "000300_benchmark.csv"
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    cached = None
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path).rename(
                columns={"日期": "date", "收盘": "close"})
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
            cached = cached.dropna().sort_values("date")
        except Exception:  # noqa: BLE001 缓存损坏视为不存在
            cached = None
        if cached is not None and not cached.empty:
            margin = pd.Timedelta(days=10)  # 覆盖判定给节假日宽限
            if cached["date"].min() <= start + margin and \
                    cached["date"].max() >= end:
                return cached, f"本地CSV {cache_path}"
        if args.offline and cached is not None and not cached.empty:
            return cached, f"本地CSV {cache_path}（覆盖不足，降级使用）"
    if args.offline:
        raise RuntimeError("offline 模式且无沪深300缓存")

    bench, source = load_benchmark(args.start, args.end, False, args.retries)
    bench = bench[["date", "close"]].copy()
    if cached is not None and not cached.empty:
        bench = pd.concat([cached, bench]).drop_duplicates("date") \
            .sort_values("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return bench, f"{source}（已缓存 {cache_path}）"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass

    args = parse_args(argv)
    command = args.command or "backtest"

    if command == "selftest":
        self_test()
        return 0
    if command == "status":
        return cmd_status()

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(90)  # 上游限流时请求可能很慢，兜底防挂起
    try:
        if command == "fetch":
            universe = fetch_universe(offline=False)
            return run_fetch(universe, args.fetch_start, args.fetch_end,
                             "data/daily", args.sleep, args.retries,
                             args.max_stocks, offline=False)

        # backtest
        universe = fetch_universe(offline=args.offline,
                                  retries=args.retries)
        symbols = universe["symbol"].tolist()
        dates, close, open_ = build_panel(symbols, args.datadir, args.adjust,
                                          PANEL_START, args.end)
        if len(dates) == 0 or close.empty:
            print("[error] 面板为空：请先运行 fetch 拉取日线", file=sys.stderr)
            return 1
        print(f"[ok] 面板: {len(dates)} 个交易日 × {close.shape[1]} 只股票"
              f"（股票池 {len(symbols)} 只）")

        bench, bench_source = _load_benchmark_cached(args, "data/daily")
        print(f"[ok] 沪深300基准: {bench_source}，{len(bench)} 行")

        out = run_backtest(dates, close, open_, symbols, args, bench)
        write_outputs(out, args)
        print(f"\n⚠ {DISCLAIMER}")
        return 0
    finally:
        socket.setdefaulttimeout(old_timeout)


if __name__ == "__main__":
    sys.exit(main())
