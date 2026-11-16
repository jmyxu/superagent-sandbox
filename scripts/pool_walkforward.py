"""n_day_low_reversal 全池 walk-forward 参数优化（issue #33 任务 2）。

背景
----
n_day_low_reversal（新低反转，TSP 开源策略种子）在 TSP 原生引擎的全市场
回测中表现出正收益。本脚本在**本仓库自己的统一口径**下（t 日收盘信号 →
t+1 日开盘成交、佣金万三 + 卖出印花税万五、小数股全仓学术口径）对其做
稳健优化：参数网格 × 按年锚定式 walk-forward（与第 4 步 optimize.py 同一
fold 结构），输出每折样本内选出的最优参数与其样本外表现，并给出一组
稳健参数（跨折样本外表现最优）在全样本期的表现。

股票池与数据：全 A 市场约 5900 只（TSP instruments 快照），日线来自
scripts/screener.py fetch --pool full_a --source tsp 导入的本地 CSV
（data/daily/<代码>_qfq.csv，由本机 TSP parquet + adj_factor 换算，
2019-07-19 ~ 2026-07-13，约 7 年）。

参数网格（4 维 × 4 档 = 256 组合，含 TSP 种子默认参数 60/1.5/-0.06/15）：
  n_days        ∈ {20, 40, 60, 120}      新低窗口（交易日）
  vol_ratio_min ∈ {1.0, 1.5, 2.0, 3.0}   最低量比（当日量/前5日均量）
  stop_loss     ∈ {0(关闭), -0.03, -0.06, -0.10}  止损线（相对入场价）
  max_hold_days ∈ {5, 10, 15, 20}        最长持有交易日
其余策略参数固定为 TSP 种子默认：require_n_day_low=True（触及 N 日收盘
新低）、require_bullish_candle=True（收阳）、use_volume_filter=True
（量比过滤）、exit_ma=20（跌破 20 日均线离场）——这三个入场条件与均线
离场是策略的"身份"，不作为网格维度。

评估口径（矩阵引擎，与逐票引擎严格等价，selftest 校验）：
  - 每只股票独立账户（初始 1.0），t 日收盘信号 t+1 日开盘成交；
  - 买入净价 open×(1+佣金)、卖出净价 open×(1-佣金-印花税)；
  - 小数股全仓：买入投入全部现金、卖出清仓（学术口径，研究信号本身）；
  - 停牌/未上市日不可成交（开盘 NaN 屏蔽交易），收盘按个股自身前向填充，
    持有股数冻结；指标在个股自身交易日上计算；
  - 窗口指标与 scripts/backtest.py compute_metrics 完全一致：
    总收益 = 窗口末净值/窗口首交易日净值 - 1（首日成交含在内），
    年化 = 总收益^(252/个股窗口内交易日数) - 1，回撤在净值曲线上取；
  - 期末仍持仓按窗口末日收盘强平（计一次交易与胜率，净值扣卖出成本）；
  - 全池指标 = 逐票指标横截面均值/中位数（0 交易票的胜率 NaN 不参与均值）。

walk-forward 折（锚定式，样本内逐年加长、样本外各 1 年不重叠）：
  fold1: 样本内 2020~2022 选参 → 样本外 2023 年验证
  fold2: 样本内 2020~2023 选参 → 样本外 2024 年验证
  fold3: 样本内 2020~2024 选参 → 样本外 2025 年验证
  fold4: 样本内 2020~2025 选参 → 样本外 2026 年验证（部分年，数据至 2026-07-13）
每折按样本内**全池平均总收益**选优；全部组合都在样本外重跑以计算
样本内-样本外秩相关（Spearman ρ）、样本内最优的样本外分位与翻转率
（过拟合诊断，与第 4 步 optimize.py 同款口径）。

已知局限（如实声明）：
  1. 幸存者偏差：股票池为当前 instruments 快照，回测期内退市股票不在池内；
  2. 数据起点 2019-07-19：n_days=120 的组合在 2020 年首个交易日约有 5 个
     交易日的热身缺口（信号 NaN，不交易），影响可忽略；
  3. 未建模滑点与冲击成本；止损以收盘价判定、次日开盘成交，跳空风险保留；
  4. 2026 年样本外仅覆盖至 2026-07-13（TSP 采集中断），为部分年。

用法：
    python scripts/pool_walkforward.py selftest   # 矩阵引擎 vs 逐票引擎等价性自检
    python scripts/pool_walkforward.py            # 全池 walk-forward 优化
    python scripts/pool_walkforward.py --grid ndlr_small   # 小网格（调试）

输出（output/optimize/n_day_low_reversal_pool/）：
  grid_pool.csv            256 组合 × 4 折 的样本内/样本外全池指标与排名
  walkforward.csv          折级选参结果（样本内最优参数、其样本外表现与分位）
  diagnostics.csv          过拟合诊断（ρ、分位、翻转率、邻点粗糙度）
  robust_fullperiod.csv    稳健参数与 TSP 默认参数的全样本期表现对比
  pool_mean_equity_*.csv   稳健/默认参数的全池等权平均净值曲线
  heatmap_pool.png         样本内 vs 样本外年化收益热力图（n_days × vol_ratio_min）
  scatter_pool.png         收益-回撤散点图（样本内/样本外）
  robust_compare.png       稳健参数 vs 默认参数的全池平均净值对比

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/pool_walkforward.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    _setup_cjk_font,
    run_backtest,
    standardize_stock,
)
from optimize import build_folds, neighbor_roughness  # noqa: E402
from screener import fetch_universe_full_a  # noqa: E402
from strategies import create_strategy  # noqa: E402

START_DEFAULT = "20200101"     # 回测整段起点（数据自 2019-07-19 热身）
END_DEFAULT = "20261231"       # 折定义终点；实际数据至 2026-07-13 截断
DATADIR_DEFAULT = "data/daily"
OUTDIR_DEFAULT = "output/optimize/n_day_low_reversal_pool"
TRAIN_YEARS_DEFAULT = 3
TEST_YEARS_DEFAULT = 1
MIN_BARS = 60                 # 参与池的股票最少要有这么多根 K 线（排除数据残缺票）

# 参数网格（4 维 × 4 档 = 256 组合，含 TSP 种子默认 60/1.5/-0.06/15）
GRID: dict[str, list] = {
    "n_days": [20, 40, 60, 120],
    "vol_ratio_min": [1.0, 1.5, 2.0, 3.0],
    "stop_loss": [0.0, -0.03, -0.06, -0.10],
    "max_hold_days": [5, 10, 15, 20],
}
GRID_SMALL: dict[str, list] = {  # 调试用小网格
    "n_days": [40, 60],
    "vol_ratio_min": [1.0, 1.5],
    "stop_loss": [-0.06],
    "max_hold_days": [10, 15],
}

# 热力图坐标轴：(横轴参数, 纵轴参数)；其余维度在热力图上取均值
HEATMAP_AXES: tuple[tuple[str, str], list[str]] = (
    ("n_days", "vol_ratio_min"), ["stop_loss", "max_hold_days"])

COMMISSION = 0.0003   # 单边佣金（万三，与 backtest.py 默认一致）
STAMP = 0.0005        # 卖出印花税（万五）


def _ffill(a: np.ndarray) -> np.ndarray:
    """沿第 0 轴（时间）前向填充 NaN（numpy 向量化，替代 pandas ffill）。"""
    if not np.isnan(a).any():
        return a
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])]


def _prev_ffill(a: np.ndarray) -> np.ndarray:
    """_ffill 结果向前平移一行（第 0 行为 NaN）。"""
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = _ffill(a)[:-1]
    return out


# ---------------------------------------------------------------------------
# 面板加载：全池 CSV → 对齐矩阵 + 每只股票的有效交易日索引
# ---------------------------------------------------------------------------

class Panel:
    """全池对齐面板：并集交易日历上的 date × stock 矩阵。

    属性：
      dates: np.ndarray (T,) datetime64[ns]，全池交易日并集；
      symbols: list[str]，列名（股票代码）；
      open/close/volume: (T, N) float32；close 在停牌/未上市日 NaN
        （此处**不**前向填充，填充只发生在构建特征与回测取值时）；
      valid: (T, N) bool：当日该股票有真实 K 线（开盘与收盘均有限）；
      valid_idx: list[np.ndarray]，每只股票的有效交易日下标；
      first_bar: np.ndarray (N,)，每只股票首根有效 K 线的日历下标；
      n_bars: np.ndarray (N,)，每只股票的有效 K 线总数。
    """

    def __init__(self, dates: np.ndarray, symbols: list[str],
                 matrices: dict[str, np.ndarray], valid: np.ndarray,
                 valid_idx: list[np.ndarray]) -> None:
        self.dates = dates
        self.symbols = symbols
        self.n_stocks = len(symbols)
        for key, mat in matrices.items():
            setattr(self, key, mat)
        self.valid = valid
        self.valid_idx = valid_idx
        self.first_bar = np.array([int(v[0]) for v in valid_idx], dtype=np.int64)
        self.n_bars = np.array([len(v) for v in valid_idx], dtype=np.int64)
        # 估值用前向填充收盘（组合无关，预计算一次）
        self.close_ff = _ffill(self.close)
        self.prev_close_ff = _prev_ffill(self.close)

    def first_valid_at_least(self, t: int) -> np.ndarray:
        """每只股票首个 ≥ t 的有效交易日下标（无则返回 T 哨兵）。"""
        out = np.full(self.n_stocks, len(self.dates), dtype=np.int64)
        for i, v in enumerate(self.valid_idx):
            j = int(np.searchsorted(v, t))
            if j < len(v):
                out[i] = v[j]
        return out

    def n_valid_between(self, lo: int, hi: int) -> np.ndarray:
        """每只股票在 [lo, hi] 内的有效交易日数（含端点）。"""
        out = np.zeros(self.n_stocks, dtype=np.int64)
        for i, v in enumerate(self.valid_idx):
            out[i] = int(np.searchsorted(v, hi + 1) - np.searchsorted(v, lo))
        return out


def _align_series(symbol: str, own_dates: np.ndarray, values: np.ndarray,
                  date_pos: dict[pd.Timestamp, int], t: int
                  ) -> np.ndarray:
    """把个股自身交易日上的数值对齐到并集日历（无 K 线的日期为 NaN）。"""
    out = np.full(t, np.nan, dtype=np.float32)
    for d, val in zip(own_dates, values):
        out[date_pos[d]] = float(val)
    return out


def load_panel(datadir: str, universe: pd.DataFrame, end: str) -> Panel:
    """读全池 CSV → 对齐面板。只保留有数据且 K 线数 ≥ MIN_BARS 的股票。"""
    datadir = Path(datadir)
    end_ts = pd.Timestamp(end)
    per_stock: dict[str, dict[str, np.ndarray]] = {}
    per_dates: dict[str, np.ndarray] = {}
    skipped_missing = 0
    for symbol in universe["symbol"]:
        csv_path = datadir / f"{symbol}_qfq.csv"
        if not csv_path.exists():
            skipped_missing += 1
            continue
        try:
            df = standardize_stock(pd.read_csv(csv_path))
        except Exception as exc:  # noqa: BLE001 单票损坏跳过
            print(f"[warn] {symbol}: CSV 读取失败，跳过: {exc}", file=sys.stderr)
            continue
        df = df[df["date"] <= end_ts]
        if len(df) < MIN_BARS:
            continue
        per_stock[symbol] = {
            "open": df["open"].to_numpy(dtype=np.float32),
            "close": df["close"].to_numpy(dtype=np.float32),
            "volume": df["volume"].to_numpy(dtype=np.float32) if "volume" in df.columns
            else np.full(len(df), np.nan, dtype=np.float32),
        }
        per_dates[symbol] = df["date"].to_numpy()

    symbols = sorted(per_stock)
    if not symbols:
        raise RuntimeError(f"{datadir} 下无可用全池 CSV（请先运行 "
                           "screener.py fetch --pool full_a --source tsp）")
    all_dates = sorted({d for arr in per_dates.values()
                        for d in np.unique(arr)})
    dates = np.array(all_dates)
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    t = len(dates)
    print(f"[ok] 面板: {t} 个交易日 × {len(symbols)} 只（缺 CSV 跳过 "
          f"{skipped_missing} 只）")

    mats: dict[str, np.ndarray] = {}
    for field in ("open", "close", "volume"):
        mat = np.full((t, len(symbols)), np.nan, dtype=np.float32)
        for j, symbol in enumerate(symbols):
            mat[:, j] = _align_series(symbol, per_dates[symbol],
                                      per_stock[symbol][field], date_pos, t)
        mats[field] = mat
    valid = np.isfinite(mats["open"]) & np.isfinite(mats["close"])
    valid_idx = [np.flatnonzero(valid[:, j]) for j in range(len(symbols))]
    return Panel(dates, symbols, mats, valid, valid_idx)


# ---------------------------------------------------------------------------
# 特征：在个股自身交易日上计算（跳过停牌日），再对齐到并集日历
# ---------------------------------------------------------------------------

def build_features(panel: Panel, n_days_values: list[int], exit_ma: int
                   ) -> dict[str, np.ndarray]:
    """预计算全部网格共用的特征矩阵（(T, N) float32，无 K 线的日期为 NaN）。

    返回 {"low_<n>": ...（各新低窗口的收盘价滚动最小值，含当日）,
          "ma_exit": ...（离场均线）, "vol_ratio": ...（当日量/前5日均量）}。
    全部在个股自身交易日序列上滚动（与逐票引擎一致，停牌日不参与窗口）。
    """
    t, n = panel.close.shape
    out: dict[str, np.ndarray] = {}
    for key in (["ma_exit", "vol_ratio"]
                + [f"low_{w}" for w in n_days_values]):
        out[key] = np.full((t, n), np.nan, dtype=np.float32)

    date_pos = {pd.Timestamp(d): i for i, d in enumerate(panel.dates)}
    for j, symbol in enumerate(panel.symbols):
        idx = panel.valid_idx[j]
        if len(idx) == 0:
            continue
        close = panel.close[idx, j]
        volume = panel.volume[idx, j]
        cs = pd.Series(close, dtype="float64")
        vs = pd.Series(volume, dtype="float64")
        for w in n_days_values:
            if len(idx) >= w:
                out[f"low_{w}"][idx[w - 1:], j] = cs.rolling(w).min().values[w - 1:]
        out["ma_exit"][idx[exit_ma - 1:], j] = \
            cs.rolling(exit_ma).mean().values[exit_ma - 1:]
        vol_prev_mean = vs.rolling(5).mean().shift(1).values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = volume / vol_prev_mean
        out["vol_ratio"][idx, j] = ratio.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 矩阵引擎：仓位状态机 + 折窗口净值/交易统计（与逐票引擎等价）
# ---------------------------------------------------------------------------

def run_positions_matrix(panel: Panel, feat: dict[str, np.ndarray],
                         n_days: int, vol_min: float, stop: float,
                         max_hold: int, exit_ma: int) -> np.ndarray:
    """在并集日历上为全部股票同时生成目标仓位矩阵 P (T, N)（0/1/NaN）。

    与 strategies.NDayLowReversalStrategy.positions 逐票语义一致：
    入场 = 触及 N 日收盘新低 且 收阳 且 量比 ≥ vol_min（t 日收盘判定）；
    离场 = 跌破 exit_ma 均线 / 收盘 ≤ 入场参考价×(1+stop) / 持有到期。
    入场参考价 = 信号次日开盘（无则当日收盘）；NaN 表示数据未就绪。
    均线"跌破"的前一日比较用个股自身上一根 K 线（停牌日跳过，前向对齐），
    与逐票引擎（个股自身日期序列）完全一致。
    """
    t, n = panel.close.shape
    close, open_ = panel.close, panel.open
    valid = panel.valid
    low_nd = feat[f"low_{n_days}"]
    ma_e = feat["ma_exit"]
    vol_ratio = feat["vol_ratio"]
    ready = np.isfinite(low_nd) & np.isfinite(ma_e) \
        & np.isfinite(vol_ratio) & valid

    close_ff = panel.close_ff
    prev_close_ff = panel.prev_close_ff
    ma_ff = _ffill(ma_e)
    prev_ma_ff = _prev_ffill(ma_e)

    p = np.full((t, n), np.nan, dtype=np.float32)
    state = np.zeros(n, dtype=bool)          # t-1 日收盘后是否持仓
    entry_ref = np.full(n, np.nan, dtype=np.float32)  # 入场参考价（次日开盘）
    entry_t = np.full(n, -1, dtype=np.int64)          # 入场信号日历下标

    open_next = np.empty_like(open_)
    open_next[:-1] = open_[1:]
    open_next[-1] = np.nan

    for t_i in range(t):
        # --- 入场（state==0 才考虑；入场当日不再检查离场） ---
        entry_cond = (~state & ready[t_i]
                      & (close[t_i] <= low_nd[t_i])
                      & (close[t_i] > open_[t_i])
                      & (vol_ratio[t_i] >= vol_min))
        state = state | entry_cond
        # 入场参考价：信号次日开盘；无则退化为当日收盘
        ref_now = np.where(np.isfinite(open_next[t_i]), open_next[t_i],
                           close[t_i]).astype(np.float32)
        entry_ref = np.where(entry_cond, ref_now, entry_ref)
        entry_t = np.where(entry_cond, t_i, entry_t)

        # --- 离场（沿用 t-1 持仓状态；停牌/无 K 线日不检查） ---
        stop_hit = np.zeros(n, dtype=bool)
        if stop < 0:
            stop_hit = (state & valid[t_i]
                        & (close[t_i] <= entry_ref * (1.0 + stop)))
        ma_break = np.zeros(n, dtype=bool)
        if t_i > 0:
            ma_break = (state & valid[t_i]
                        & (close[t_i] < ma_e[t_i])
                        & (prev_close_ff[t_i] >= prev_ma_ff[t_i]))
        hold_hit = np.zeros(n, dtype=bool)
        if max_hold > 0:
            hold_hit = state & valid[t_i] & (t_i - entry_t >= max_hold)
        exited = stop_hit | ma_break | hold_hit
        state = state & ~exited

        p[t_i] = np.where(ready[t_i], state.astype(np.float32), np.nan)
    return p


def _build_position_edges(panel: Panel, p: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """预计算边沿判定用的仓位视图（每组合一次，避免逐窗口重复）：

    p_ff    = 仓位前向填充（t-1 取个股最后有信号日的仓位）；
    p2_own  = 个股倒数第二根 K 线的仓位（t-1 停牌时不能用日历前向填充）。
    """
    p_ff = _ffill(p.astype(np.float32))
    p2_own = np.full_like(p, np.nan)
    for j, idx in enumerate(panel.valid_idx):
        if len(idx) >= 2:
            p2_own[idx[2:], j] = p[idx[:-2], j]
    return p_ff, p2_own


def _window_metrics(panel: Panel, p_ff: np.ndarray, p2_own: np.ndarray,
                    s: int, e: int) -> dict[str, np.ndarray]:
    """窗口 [s, e]（日历下标，含端点）上的逐票指标。

    等价于 backtest.run_backtest(df, strategy, start, end, fractional=True)
    的 compute_metrics + trade_stats：
      窗口起始状态为空仓；t 日边沿（个股自身 t-1/t-2 根 K 线的仓位变化）
      在 t 日开盘成交；停牌/无开盘日不可成交；期末仍持仓按窗口内最后一根
      K 线收盘强平。
    返回 dict：total / annual / mdd / trades / wins / win_rate / n_traded，
    均为 (N,) 数组；个股窗口内无 K 线时为 NaN（不参与全池统计）。
    """
    n = panel.n_stocks
    close, open_ = panel.close, panel.open
    valid = panel.valid
    close_ff = panel.close_ff
    prev_close_ff = panel.prev_close_ff
    buy_fee = 1.0 + COMMISSION
    sell_fee = 1.0 - COMMISSION - STAMP

    eq = np.ones(n, dtype=np.float32)          # 窗口净值（窗口起点=1）
    base = np.full(n, np.nan, dtype=np.float32)  # 窗口首根 K 线收盘净值
    peak = np.ones(n, dtype=np.float32)
    mdd = np.zeros(n, dtype=np.float32)
    state = np.zeros(n, dtype=bool)            # 窗口内 t-1 日收盘是否持仓
    entry_px = np.full(n, np.nan, dtype=np.float32)
    trades = np.zeros(n, dtype=np.int64)
    wins = np.zeros(n, dtype=np.int64)
    first_seen = np.zeros(n, dtype=bool)

    for t_i in range(s, e + 1):
        # 引擎允许成交的最早日是 s+1（要求 t 与 t-1 都在窗口内）
        p1 = p_ff[t_i - 1] if t_i - 1 >= 0 else np.full(n, np.nan)
        p2 = p2_own[t_i]
        buy = (~state & (p1 == 1) & (p2 == 0)
               & np.isfinite(open_[t_i]) & (t_i > s))
        sell = (state & (p1 == 0) & (p2 == 1)
                & np.isfinite(open_[t_i]) & (t_i > s))

        # 日收益乘数：持仓 → 收盘对收盘；买卖日按开盘净价成交
        m_t = np.ones(n, dtype=np.float32)
        m_t = np.where(state, close_ff[t_i] / prev_close_ff[t_i], m_t)
        m_t = np.where(buy, close[t_i] / (open_[t_i] * buy_fee), m_t)
        m_t = np.where(sell, (open_[t_i] * sell_fee) / prev_close_ff[t_i], m_t)
        eq = eq * m_t

        # 交易统计（成交净价口径，与 Trade.net_return 一致）
        entry_px = np.where(buy, open_[t_i] * buy_fee, entry_px)
        exit_px = open_[t_i] * sell_fee
        ret = exit_px / entry_px - 1.0
        traded = sell & np.isfinite(entry_px)
        trades = trades + traded.astype(np.int64)
        wins = wins + (traded & (ret > 0)).astype(np.int64)

        state = (state | buy) & ~sell

        # 回撤口径与引擎一致：从个股窗口首根 K 线的净值起算
        newly = valid[t_i] & ~first_seen
        base = np.where(newly, eq, base)
        peak = np.where(newly, eq, peak)
        first_seen = first_seen | valid[t_i]
        peak = np.maximum(peak, eq)
        mdd = np.minimum(mdd, eq / peak - 1.0)

    # 期末强平：仍持仓的按窗口内最后一根 K 线收盘净价卖出，净值相应调整
    liq = state & np.isfinite(entry_px)
    eq = eq * np.where(liq, sell_fee, 1.0)
    ret_liq = close_ff[e] * sell_fee / entry_px - 1.0
    trades = trades + liq.astype(np.int64)
    wins = wins + (liq & (ret_liq > 0)).astype(np.int64)
    peak = np.maximum(peak, eq)
    mdd = np.minimum(mdd, eq / peak - 1.0)

    in_win = first_seen
    total = np.full(n, np.nan, dtype=np.float32)
    total[in_win] = eq[in_win] / base[in_win] - 1.0
    mdd[~in_win] = np.nan

    n_bars = panel.n_valid_between(s, e)
    annual = np.full(n, np.nan, dtype=np.float32)
    ok = in_win & (n_bars > 1) & (total >= -1.0)
    annual[ok] = np.power(1.0 + total[ok], 252.0 / n_bars[ok]) - 1.0

    win_rate = np.full(n, np.nan, dtype=np.float32)
    has_trades = trades > 0
    win_rate[has_trades] = wins[has_trades] / trades[has_trades]
    return {
        "total": total, "annual": annual, "mdd": mdd,
        "trades": trades.astype(np.float32), "wins": wins.astype(np.float32),
        "win_rate": win_rate, "n_traded": in_win & has_trades,
    }


def _pool_stats(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """逐票指标 → 全池横截面统计（0 交易票的胜率 NaN 不参与均值）。"""
    total = metrics["total"]
    out: dict[str, float] = {
        "pool_total": float(np.nanmean(total)),
        "pool_median_total": float(np.nanmedian(total)),
        "pool_pct_positive": float(np.nanmean(total > 0)),
        "pool_annual": float(np.nanmean(metrics["annual"])),
        "pool_mdd": float(np.nanmean(metrics["mdd"])),
        "pool_mean_trades": float(np.nanmean(metrics["trades"])),
        "pool_n_traded": float(np.nansum(metrics["n_traded"])),
    }
    wr = metrics["win_rate"]
    out["pool_win_rate"] = float(np.nanmean(wr)) if np.isfinite(wr).any() else np.nan
    return out


# ---------------------------------------------------------------------------
# walk-forward 主流程
# ---------------------------------------------------------------------------

def grid_combos(grid: dict) -> list[tuple[dict, object]]:
    """生成网格内全部合法组合（经策略构造校验）。"""
    keys = list(grid)
    out: list[tuple[dict, object]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        try:
            strat = create_strategy("n_day_low_reversal", **combo)
        except ValueError:
            continue
        out.append((combo, strat))
    return out


def fmt_date(ts: str) -> str:
    """YYYYMMDD → YYYY-MM-DD。"""
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"


def run_walkforward(panel: Panel, feat: dict[str, np.ndarray],
                    folds: list[tuple[str, str, str, str]],
                    grid: dict, exit_ma: int,
                    outdir: Path) -> dict:
    """全池 walk-forward：网格 × 折 的样本内/样本外全池指标、选参、诊断。"""
    combos = grid_combos(grid)
    keys = list(grid)
    n_folds = len(folds)
    t_start = time.time()
    print(f"walk-forward: {len(combos)} 组合 × {n_folds} 折 × "
          f"{panel.n_stocks} 只，矩阵引擎开始")

    # 折窗口（日历下标）
    fold_win: list[tuple[int, int, int, int]] = []
    for tr_s, tr_e, te_s, te_e in folds:
        tr_lo = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(tr_s))))
        tr_hi = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(tr_e)),
                                    side="right")) - 1
        te_lo = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(te_s))))
        te_hi = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(te_e)),
                                    side="right")) - 1
        fold_win.append((tr_lo, tr_hi, te_lo, te_hi))

    rows: list[dict] = []
    for ci, (combo, _strat) in enumerate(combos):
        p = run_positions_matrix(panel, feat, int(combo["n_days"]),
                                 float(combo["vol_ratio_min"]),
                                 float(combo["stop_loss"]),
                                 int(combo["max_hold_days"]), exit_ma)
        p_ff, p2_own = _build_position_edges(panel, p)
        row: dict = {"strategy": "n_day_low_reversal"}
        row.update(combo)
        for fi, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(fold_win, 1):
            is_m = _window_metrics(panel, p_ff, p2_own, tr_lo, tr_hi)
            oos_m = _window_metrics(panel, p_ff, p2_own, te_lo, te_hi)
            is_s = _pool_stats(is_m)
            oos_s = _pool_stats(oos_m)
            row.update({
                f"fold{fi}_is_total": is_s["pool_total"],
                f"fold{fi}_is_annual": is_s["pool_annual"],
                f"fold{fi}_is_mdd": is_s["pool_mdd"],
                f"fold{fi}_is_trades": is_s["pool_mean_trades"],
                f"fold{fi}_is_win_rate": is_s["pool_win_rate"],
                f"fold{fi}_oos_total": oos_s["pool_total"],
                f"fold{fi}_oos_annual": oos_s["pool_annual"],
                f"fold{fi}_oos_mdd": oos_s["pool_mdd"],
                f"fold{fi}_oos_trades": oos_s["pool_mean_trades"],
                f"fold{fi}_oos_win_rate": oos_s["pool_win_rate"],
                f"fold{fi}_oos_pct_positive": oos_s["pool_pct_positive"],
            })
        rows.append(row)
        if (ci + 1) % 16 == 0 or ci + 1 == len(combos):
            el = time.time() - t_start
            print(f"[进度] {ci + 1}/{len(combos)} 组合，用时 {el:.0f}s", flush=True)

    g = pd.DataFrame(rows)
    for fi in range(1, n_folds + 1):
        g[f"fold{fi}_is_rank"] = g[f"fold{fi}_is_total"].rank(ascending=False, method="min")
        g[f"fold{fi}_oos_rank"] = g[f"fold{fi}_oos_total"].rank(ascending=False, method="min")
    for col in ("total", "annual", "mdd", "trades", "win_rate"):
        g[f"mean_is_{col}"] = g[[f"fold{fi}_is_{col}" for fi in range(1, n_folds + 1)]].mean(axis=1)
        g[f"mean_oos_{col}"] = g[[f"fold{fi}_oos_{col}" for fi in range(1, n_folds + 1)]].mean(axis=1)
    g["mean_oos_rank"] = g[[f"fold{fi}_oos_rank" for fi in range(1, n_folds + 1)]].mean(axis=1)

    n_combos = len(g)
    combo_index = {tuple(r[k] for k in keys): i for i, r in g.iterrows()}
    default_params = {k: v.default for k, v in
                      create_strategy("n_day_low_reversal").params.items()}
    default_combo = {k: default_params[k] for k in keys}
    default_idx = combo_index[tuple(default_combo[k] for k in keys)]

    # ---------- 诊断 ----------
    rho = float(g["mean_is_total"].rank(method="average")
                .corr(g["mean_oos_total"].rank(method="average")))
    is_best_idx = int(g["mean_is_total"].idxmax())
    robust_idx = int(g.sort_values(["mean_oos_annual", "mean_is_annual"],
                                   ascending=[False, False]).index[0])
    is_best_combo = {k: g.at[is_best_idx, k] for k in keys}
    robust_combo = {k: g.at[robust_idx, k] for k in keys}
    is_topq = g["mean_is_total"] >= g["mean_is_total"].quantile(0.75)
    oos_botq = g["mean_oos_total"] <= g["mean_oos_total"].quantile(0.25)
    flip_rate = float((is_topq & oos_botq).mean())

    x_axis, y_axis = HEATMAP_AXES
    surf_is = g.pivot_table(index=y_axis, columns=x_axis, values="mean_is_annual",
                            aggfunc="mean").sort_index(ascending=True)
    surf_oos = g.pivot_table(index=y_axis, columns=x_axis, values="mean_oos_annual",
                             aggfunc="mean").sort_index(ascending=True)
    rough_is = neighbor_roughness(surf_is)
    rough_oos = neighbor_roughness(surf_oos)

    # ---------- 折级选参 ----------
    wf_rows: list[dict] = []
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        sel = g.loc[g[f"fold{fi}_is_total"].idxmax()]
        best = g.loc[g[f"fold{fi}_oos_total"].idxmax()]
        def_col = g.at[default_idx, f"fold{fi}_oos_total"]
        wf_rows.append({
            "fold": fi,
            "train_start": fmt_date(tr_s), "train_end": fmt_date(tr_e),
            "test_start": fmt_date(te_s), "test_end": fmt_date(te_e),
            "n_combos": n_combos,
            "is_best_params": ",".join(f"{k}={sel[k]}" for k in keys),
            "is_best_is_total": sel[f"fold{fi}_is_total"],
            "is_best_is_annual": sel[f"fold{fi}_is_annual"],
            "is_best_oos_total": sel[f"fold{fi}_oos_total"],
            "is_best_oos_annual": sel[f"fold{fi}_oos_annual"],
            "is_best_oos_rank": int(sel[f"fold{fi}_oos_rank"]),
            "is_best_oos_pct": sel[f"fold{fi}_oos_rank"] / n_combos,
            "oos_best_params": ",".join(f"{k}={best[k]}" for k in keys),
            "oos_best_oos_total": best[f"fold{fi}_oos_total"],
            "oos_best_is_total": best[f"fold{fi}_is_total"],
            "default_oos_total": def_col,
            "grid_mean_oos_total": g[f"fold{fi}_oos_total"].mean(),
            "grid_median_oos_total": g[f"fold{fi}_oos_total"].median(),
        })
    wf = pd.DataFrame(wf_rows)
    is_best_mean_oos = float(wf["is_best_oos_total"].mean())
    is_best_mean_oos_pct = float(wf["is_best_oos_pct"].mean())

    # ---------- 落盘 ----------
    outdir.mkdir(parents=True, exist_ok=True)
    grid_cols = (["strategy"] + keys
                 + [f"fold{fi}_{side}_{m}" for fi in range(1, n_folds + 1)
                    for side in ("is", "oos")
                    for m in ("total", "annual", "mdd", "trades", "win_rate")]
                 + [f"fold{fi}_oos_pct_positive" for fi in range(1, n_folds + 1)]
                 + [f"fold{fi}_{side}_rank" for fi in range(1, n_folds + 1)
                    for side in ("is", "oos")]
                 + ["mean_is_total", "mean_is_annual", "mean_is_mdd",
                    "mean_oos_total", "mean_oos_annual", "mean_oos_mdd",
                    "mean_oos_rank"])
    g[grid_cols].to_csv(outdir / "grid_pool.csv", index=False, encoding="utf-8-sig")
    wf.to_csv(outdir / "walkforward.csv", index=False, encoding="utf-8-sig")

    robust_row = g.loc[robust_idx]
    default_row = g.loc[default_idx]
    diag = pd.DataFrame([
        ("n_combos", n_combos), ("n_folds", n_folds),
        ("n_stocks", panel.n_stocks),
        ("data_start", str(panel.dates[0])[:10]),
        ("data_end", str(panel.dates[-1])[:10]),
        ("spearman_rho_is_oos", rho),
        ("is_best_params", ",".join(f"{k}={is_best_combo[k]}" for k in keys)),
        ("is_best_mean_is_annual", g.at[is_best_idx, "mean_is_annual"]),
        ("is_best_mean_oos_total", g.at[is_best_idx, "mean_oos_total"]),
        ("is_best_mean_oos_annual", g.at[is_best_idx, "mean_oos_annual"]),
        ("is_best_mean_oos_pct", is_best_mean_oos_pct),
        ("robust_params", ",".join(f"{k}={robust_combo[k]}" for k in keys)),
        ("robust_mean_is_annual", robust_row["mean_is_annual"]),
        ("robust_mean_oos_annual", robust_row["mean_oos_annual"]),
        ("robust_mean_oos_rank", robust_row["mean_oos_rank"]),
        ("default_params", ",".join(f"{k}={default_combo[k]}" for k in keys)),
        ("default_mean_is_annual", default_row["mean_is_annual"]),
        ("default_mean_oos_annual", default_row["mean_oos_annual"]),
        ("robust_vs_default_oos_delta", robust_row["mean_oos_annual"]
         - default_row["mean_oos_annual"]),
        ("flip_rate_is_topq_to_oos_botq", flip_rate),
        ("roughness_is_annual", rough_is),
        ("roughness_oos_annual", rough_oos),
        ("disclaimer", DISCLAIMER),
    ], columns=["metric", "value"])
    diag.to_csv(outdir / "diagnostics.csv", index=False, encoding="utf-8-sig")

    _pct = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x * 100:.2f}%"
    print(f"\n===== 全池 walk-forward: n_day_low_reversal "
          f"（{len(combos)} 组合 × {n_folds} 折 × {panel.n_stocks} 只）=====")
    print(wf[["fold", "is_best_params", "is_best_is_total", "is_best_oos_total",
              "is_best_oos_rank", "oos_best_params", "default_oos_total"]].assign(
        is_best_is_total=lambda d: d["is_best_is_total"].map(_pct),
        is_best_oos_total=lambda d: d["is_best_oos_total"].map(_pct),
        default_oos_total=lambda d: d["default_oos_total"].map(_pct),
    ).to_string(index=False))
    print(f"Spearman ρ(样本内均值, 样本外均值) = {rho:+.3f}")
    print(f"样本内最优参数 {','.join(f'{k}={is_best_combo[k]}' for k in keys)}："
          f"样本外均值 {_pct(g.at[is_best_idx, 'mean_oos_annual'])}，"
          f"样本外平均分位 {is_best_mean_oos_pct:.0%}")
    print(f"稳健参数 {','.join(f'{k}={robust_combo[k]}' for k in keys)}："
          f"样本外年化均值 {_pct(robust_row['mean_oos_annual'])}"
          f"（默认参数 {_pct(default_row['mean_oos_annual'])}），"
          f"样本外平均排名 {robust_row['mean_oos_rank']:.1f}/{n_combos}")
    print(f"翻转率 {flip_rate:.0%}；邻点粗糙度：样本内 {_pct(rough_is)}/步，"
          f"样本外 {_pct(rough_oos)}/步")
    print(f"⚠ {DISCLAIMER}")

    return {
        "grid_df": g, "wf": wf, "combos": combos, "keys": keys,
        "default_combo": default_combo, "robust_combo": robust_combo,
        "is_best_combo": is_best_combo, "rho": rho, "flip_rate": flip_rate,
        "is_best_mean_oos_pct": is_best_mean_oos_pct,
    }


# ---------------------------------------------------------------------------
# 稳健参数全样本期评估 + 绘图
# ---------------------------------------------------------------------------

def evaluate_full(panel: Panel, feat: dict[str, np.ndarray], combo: dict,
                  exit_ma: int, start: str, end: str) -> dict:
    """给定参数在全样本期 [start, end] 的逐票/全池指标与净值曲线。"""
    s = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(start))))
    e = int(np.searchsorted(panel.dates, np.datetime64(pd.Timestamp(end)),
                            side="right")) - 1
    p = run_positions_matrix(panel, feat, int(combo["n_days"]),
                             float(combo["vol_ratio_min"]),
                             float(combo["stop_loss"]),
                             int(combo["max_hold_days"]), exit_ma)
    p_ff, p2_own = _build_position_edges(panel, p)
    metrics = _window_metrics(panel, p_ff, p2_own, s, e)
    stats = _pool_stats(metrics)
    stats["params"] = ",".join(f"{k}={combo[k]}" for k in combo)
    stats["n_stocks"] = panel.n_stocks

    # 全池等权平均净值曲线（窗口起点=1，含逐日参与股票数）
    curve = _mean_equity_curve(panel, p_ff, p2_own, s, e)
    return stats, curve


def _mean_equity_curve(panel: Panel, p_ff: np.ndarray, p2_own: np.ndarray,
                       s: int, e: int) -> pd.DataFrame:
    """全池等权平均净值曲线：每只股票独立账户（窗口起点=1），逐日平均。"""
    n = panel.n_stocks
    close, open_ = panel.close, panel.open
    buy_fee = 1.0 + COMMISSION
    sell_fee = 1.0 - COMMISSION - STAMP
    close_ff = panel.close_ff
    prev_close_ff = panel.prev_close_ff

    eq = np.ones(n, dtype=np.float32)
    state = np.zeros(n, dtype=bool)
    entry_px = np.full(n, np.nan, dtype=np.float32)
    curve_rows: list[dict] = []
    listed_at = panel.first_valid_at_least(s)
    for t_i in range(s, e + 1):
        p1 = p_ff[t_i - 1] if t_i - 1 >= 0 else np.full(n, np.nan)
        p2 = p2_own[t_i]
        buy = (~state & (p1 == 1) & (p2 == 0) & np.isfinite(open_[t_i])
               & (t_i > s))
        sell = (state & (p1 == 0) & (p2 == 1) & np.isfinite(open_[t_i])
                & (t_i > s))
        entry_px = np.where(buy, open_[t_i] * buy_fee, entry_px)
        m_t = np.ones(n, dtype=np.float32)
        m_t = np.where(state, close_ff[t_i] / prev_close_ff[t_i], m_t)
        m_t = np.where(buy, close[t_i] / (open_[t_i] * buy_fee), m_t)
        m_t = np.where(sell, (open_[t_i] * sell_fee) / prev_close_ff[t_i], m_t)
        eq = eq * m_t
        state = (state | buy) & ~sell
        curve_rows.append({
            "date": pd.Timestamp(panel.dates[t_i]).strftime("%Y-%m-%d"),
            "mean_strat": float(np.nanmean(eq[listed_at <= t_i])),
            "n_stocks": int(np.sum(listed_at <= t_i)),
        })
    # 期末强平在窗口末日的净值调整（口径与 _window_metrics 一致）
    liq = state & np.isfinite(entry_px)
    if liq.any():
        eq = eq * np.where(liq, sell_fee, 1.0)
        curve_rows[-1] = {
            "date": curve_rows[-1]["date"],
            "mean_strat": float(np.nanmean(eq[listed_at <= e])),
            "n_stocks": curve_rows[-1]["n_stocks"],
        }
    return pd.DataFrame(curve_rows)


def plot_heatmap(grid_df: pd.DataFrame, robust_combo: dict,
                 is_best_combo: dict, out_png: Path) -> None:
    """样本内 vs 样本外年化收益热力图（其余维度取均值）。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    x_param, y_param = HEATMAP_AXES[0]
    surf_is = grid_df.pivot_table(index=y_param, columns=x_param,
                                  values="mean_is_annual", aggfunc="mean")
    surf_oos = grid_df.pivot_table(index=y_param, columns=x_param,
                                   values="mean_oos_annual", aggfunc="mean")
    x_labels = [str(v) for v in surf_is.columns]
    y_labels = [str(v) for v in surf_is.index]
    panels = (("样本内（各折年化收益均值 %）", surf_is),
              ("样本外（各折年化收益均值 %）", surf_oos))
    all_vals = np.concatenate([surf_is.to_numpy().ravel(),
                               surf_oos.to_numpy().ravel()])
    all_vals = all_vals[~np.isnan(all_vals)] * 100
    vmax = max(0.0, float(np.nanmax(all_vals)))
    vmin = min(0.0, float(np.nanmin(all_vals)))
    if vmin >= 0.0:
        vmin = -max(vmax * 0.1, 0.01)
    if vmax <= 0.0:
        vmax = max(-vmin * 0.1, 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (title, surf) in zip(axs, panels):
        data = surf.to_numpy(dtype=float) * 100
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(x_labels)), x_labels)
        ax.set_yticks(range(len(y_labels)), y_labels)
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(title)
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                if not masked.mask[i, j]:
                    ax.text(j, i, f"{masked[i, j]:.0f}", ha="center",
                            va="center", fontsize=8, color="black")
        for combo, marker, color in ((is_best_combo, "rect", "black"),
                                     (robust_combo, "diamond", "blue")):
            if combo[x_param] in surf.columns and combo[y_param] in surf.index:
                bx = x_labels.index(str(combo[x_param]))
                by = y_labels.index(str(combo[y_param]))
                if marker == "rect":
                    ax.add_patch(Rectangle((bx - 0.5, by - 0.5), 1, 1,
                                           fill=False, edgecolor=color, lw=2.5))
                else:
                    ax.plot(bx, by, marker="D", markersize=9, mfc="none",
                            mec=color, mew=2)
    cax = fig.add_axes([0.93, 0.14, 0.012, 0.70])
    fig.colorbar(im, cax=cax, label="年化收益率（%）")
    fig.suptitle("n_day_low_reversal 全池 walk-forward 参数热力图"
                 "（其余维度取均值，小数股全仓口径）\n"
                 "■=样本内全局最优  ◇=稳健参数（样本外均值最优）")
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 热力图已保存: {out_png}")


def plot_scatter(grid_df: pd.DataFrame, robust_combo: dict,
                 is_best_combo: dict, default_combo: dict, rho: float,
                 is_best_mean_oos_pct: float, flip_rate: float,
                 out_png: Path) -> None:
    """收益-回撤散点图：每组合一个样本内点 + 一个样本外点。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.scatter(grid_df["mean_is_annual"] * 100, grid_df["mean_is_mdd"] * 100,
               s=26, alpha=0.75, color="#1f77b4", label="样本内（折均值）")
    ax.scatter(grid_df["mean_oos_annual"] * 100, grid_df["mean_oos_mdd"] * 100,
               s=26, alpha=0.75, color="#ff7f0e", label="样本外（折均值）")
    combo_keys = ("n_days", "vol_ratio_min", "stop_loss", "max_hold_days")
    combo_index = {tuple(r[k] for k in combo_keys): i
                   for i, r in grid_df.iterrows()}

    def mark(combo: dict, style: str, label: str, color: str) -> None:
        i = combo_index[tuple(combo[k] for k in combo_keys)]
        ax.scatter([grid_df.at[i, "mean_oos_annual"] * 100],
                   [grid_df.at[i, "mean_oos_mdd"] * 100], marker=style,
                   s=130, facecolors="none", edgecolors=color, lw=2, label=label)

    is_best_i = combo_index[tuple(is_best_combo[k] for k in combo_keys)]
    ax.scatter([grid_df.at[is_best_i, "mean_is_annual"] * 100],
               [grid_df.at[is_best_i, "mean_is_mdd"] * 100], marker="s",
               s=150, facecolors="none", edgecolors="black", lw=2,
               label="样本内全局最优（■）")
    ax.annotate("", xy=(grid_df.at[is_best_i, "mean_oos_annual"] * 100,
                        grid_df.at[is_best_i, "mean_oos_mdd"] * 100),
                xytext=(grid_df.at[is_best_i, "mean_is_annual"] * 100,
                        grid_df.at[is_best_i, "mean_is_mdd"] * 100),
                arrowprops=dict(arrowstyle="->", color="black", lw=1,
                                linestyle=":"))
    mark(robust_combo, "D", "稳健参数（◇）", "darkblue")
    mark(default_combo, "^", "默认参数（△）", "darkgreen")
    ax.text(0.03, 0.03,
            f"Spearman ρ = {rho:+.2f}\n"
            f"样本内最优的平均样本外分位 = {is_best_mean_oos_pct:.0%}\n"
            f"翻转率（样本内前1/4→样本外后1/4） = {flip_rate:.0%}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_xlabel("年化收益率（%）")
    ax.set_ylabel("最大回撤（%）")
    ax.set_title("n_day_low_reversal 全池收益-回撤散点图（小数股全仓口径）\n"
                 "虚线箭头 = 样本内全局最优 → 其在样本外的表现")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 收益-回撤散点图已保存: {out_png}")


# ---------------------------------------------------------------------------
# 自检：矩阵引擎 vs 逐票引擎（backtest.run_backtest）等价性
# ---------------------------------------------------------------------------

def selftest() -> None:
    """合成数据上断言矩阵引擎与逐票引擎（小数股口径）结果一致。"""
    rng = np.random.default_rng(7)
    n_days_total = 160
    dates = pd.bdate_range("2019-06-01", periods=n_days_total)
    n_stocks = 6
    frames: dict[str, pd.DataFrame] = {}
    for k in range(n_stocks):
        symbol = f"s{k}"
        close = np.maximum(5 + np.cumsum(rng.normal(0.0005, 0.02, n_days_total)), 0.5)
        open_ = close * (1 + rng.normal(0, 0.004, n_days_total))
        volume = rng.uniform(1e5, 5e6, n_days_total)
        # 人为制造两段停牌（无 K 线）+ 一只新股（前 20 日无数据）
        df = pd.DataFrame({"date": dates, "open": np.maximum(open_, 0.01),
                           "close": np.maximum(close, 0.01), "volume": volume})
        if k == 0:
            df = df.iloc[20:].reset_index(drop=True)
        if k == 2:
            df = pd.concat([df.iloc[:60], df.iloc[70:]]).reset_index(drop=True)
        frames[symbol] = df

    universe = pd.DataFrame({"symbol": sorted(frames)})
    # 直接构造面板（不经 CSV 落盘）
    panel = _panel_from_frames(frames, universe)
    feat = build_features(panel, [20, 40, 60, 120], exit_ma=20)

    combos = [
        {"n_days": 60, "vol_ratio_min": 1.5, "stop_loss": -0.06, "max_hold_days": 15},
        {"n_days": 20, "vol_ratio_min": 1.0, "stop_loss": 0.0, "max_hold_days": 5},
        {"n_days": 120, "vol_ratio_min": 3.0, "stop_loss": -0.10, "max_hold_days": 20},
    ]
    windows = [(pd.Timestamp("2019-12-01"), pd.Timestamp("2020-06-30")),
               (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
               (pd.Timestamp("2020-02-03"), pd.Timestamp("2020-05-29")),
               (pd.Timestamp("2019-08-20"), pd.Timestamp("2019-12-31"))]

    max_err = 0.0
    for combo in combos:
        strat = create_strategy("n_day_low_reversal", **combo)
        p = run_positions_matrix(panel, feat, int(combo["n_days"]),
                                 float(combo["vol_ratio_min"]),
                                 float(combo["stop_loss"]),
                                 int(combo["max_hold_days"]), 20)
        p_ff, p2_own = _build_position_edges(panel, p)
        for ts, te in windows:
            s = int(np.searchsorted(panel.dates, np.datetime64(ts)))
            e = int(np.searchsorted(panel.dates, np.datetime64(te),
                                    side="right")) - 1
            mm = _window_metrics(panel, p_ff, p2_own, s, e)
            for j, symbol in enumerate(panel.symbols):
                df = frames[symbol]
                # 与真实管线一致：load_stock 以窗口终点截断数据（引擎期末强平
                # 即发生在窗口内最后一根 K 线）
                df = df[df["date"] <= te].reset_index(drop=True)
                eq, trades, eval_start, _ = run_backtest(
                    df, strat, ts.strftime("%Y%m%d"), te.strftime("%Y%m%d"),
                    100000.0, COMMISSION, STAMP, True)
                # 逐票引擎在窗口内无 K 线时 eval_start 落在窗口外 → 跳过对比
                window = (df["date"] >= ts) & (df["date"] <= te)
                if not window.any():
                    assert not np.isfinite(mm["total"][j]), \
                        f"{symbol}: 窗口无 K 线时矩阵引擎应返回 NaN"
                    continue
                eng_total = eq[-1] / eq[eval_start] - 1
                err = abs(eng_total - float(mm["total"][j]))
                max_err = max(max_err, err)
                assert err < 2e-3, (
                    f"{symbol} {combo} {ts}~{te}: 矩阵引擎总收益 "
                    f"{mm['total'][j]:.6f} vs 逐票引擎 {eng_total:.6f}")
                assert int(mm["trades"][j]) == len(trades), (
                    f"{symbol} {combo} {ts}~{te}: 交易数不一致 "
                    f"{int(mm['trades'][j])} vs {len(trades)}")
                wins = sum(1 for t in trades if t.net_return > 0)
                assert int(mm["wins"][j]) == wins, (
                    f"{symbol} {combo} {ts}~{te}: 盈利交易数不一致")
                if len(trades):
                    assert abs(float(mm["win_rate"][j]) - wins / len(trades)) < 1e-6
                # 年化对比（n_bars 口径）
                n_bars = len(eq) - eval_start
                eng_annual = (1 + eng_total) ** (252 / n_bars) - 1 if n_bars > 1 \
                    else np.nan
                if np.isfinite(eng_annual):
                    assert abs(float(mm["annual"][j]) - eng_annual) < 2e-3, (
                        f"{symbol} {combo} {ts}~{te}: 年化不一致 "
                        f"{mm['annual'][j]:.6f} vs {eng_annual:.6f}")
    print(f"[ok] selftest 通过：{len(combos)} 组合 × {len(windows)} 窗口 × "
          f"{n_stocks} 只（含停牌/新股缺口），矩阵引擎与逐票引擎的"
          f"总收益/年化/交易数/胜率一致（最大绝对误差 {max_err:.2e}）")


def _panel_from_frames(frames: dict[str, pd.DataFrame],
                       universe: pd.DataFrame) -> Panel:
    """selftest 用：直接把 DataFrame 字典组装成 Panel。"""
    symbols = sorted(frames)
    all_dates = sorted({d for df in frames.values() for d in df["date"]})
    dates = np.array(all_dates)
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    t = len(dates)
    mats: dict[str, np.ndarray] = {}
    for field in ("open", "close", "high", "low", "volume"):
        mat = np.full((t, len(symbols)), np.nan, dtype=np.float32)
        for j, symbol in enumerate(symbols):
            df = frames[symbol]
            if field not in df.columns:
                continue
            for d, val in zip(df["date"], df[field]):
                mat[date_pos[pd.Timestamp(d)], j] = float(val)
        mats[field] = mat
    valid = np.isfinite(mats["open"]) & np.isfinite(mats["close"])
    valid_idx = [np.flatnonzero(valid[:, j]) for j in range(len(symbols))]
    return Panel(dates, symbols, mats, valid, valid_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="n_day_low_reversal 全池 walk-forward 参数优化"
                    "（参数网格 × 按年锚定式 walk-forward，issue #33 任务 2）",
        epilog=(
            "示例:\n"
            "  python scripts/pool_walkforward.py\n"
            "  python scripts/pool_walkforward.py --grid ndlr_small\n"
            "  python scripts/pool_walkforward.py selftest\n"
            "所有输出仅为历史模拟，不构成投资建议。"
        ),
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "selftest"],
                        help="run=全池 walk-forward（默认）/ selftest=引擎等价性自检")
    parser.add_argument("--start", default=START_DEFAULT,
                        help=f"回测整段起点 YYYYMMDD，默认 {START_DEFAULT}")
    parser.add_argument("--end", default=END_DEFAULT,
                        help=f"折定义终点 YYYYMMDD，默认 {END_DEFAULT}"
                             "（实际数据至 2026-07-13 截断）")
    parser.add_argument("--train-years", type=int, default=TRAIN_YEARS_DEFAULT,
                        help=f"首个样本内窗口年数，默认 {TRAIN_YEARS_DEFAULT}")
    parser.add_argument("--test-years", type=int, default=TEST_YEARS_DEFAULT,
                        help=f"每个样本外窗口年数，默认 {TEST_YEARS_DEFAULT}")
    parser.add_argument("--grid", default="ndlr", choices=["ndlr", "ndlr_small"],
                        help="参数网格：ndlr（默认，256 组合）/ ndlr_small（调试）")
    parser.add_argument("--datadir", default=DATADIR_DEFAULT,
                        help=f"行情 CSV 目录，默认 {DATADIR_DEFAULT}")
    parser.add_argument("--outdir", default=OUTDIR_DEFAULT,
                        help=f"输出目录，默认 {OUTDIR_DEFAULT}")
    parser.add_argument("--exit-ma", type=int, default=20,
                        help="离场均线窗口（固定，不进网格），默认 20")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 图")
    args = parser.parse_args(argv)
    if args.train_years < 1 or args.test_years < 1:
        parser.error("--train-years/--test-years 必须为正整数")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass

    args = parse_args(argv)
    if args.command == "selftest":
        selftest()
        return 0

    folds = build_folds(args.start, args.end, args.train_years, args.test_years)
    if not folds:
        print(f"[error] {args.start}~{args.end} 放不下至少一个 "
              f"{args.train_years}+{args.test_years} 年的 walk-forward fold",
              file=sys.stderr)
        return 2
    grid = GRID_SMALL if args.grid == "ndlr_small" else GRID
    print(f"walk-forward 区间: {args.start} ~ {args.end} | 网格 "
          f"{args.grid}（{len(grid)} 维）| 离场均线 {args.exit_ma}")
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        print(f"  fold{fi}: 样本内 {fmt_date(tr_s)}~{fmt_date(tr_e)} 选参 → "
              f"样本外 {fmt_date(te_s)}~{fmt_date(te_e)} 验证")
    print(f"⚠ {DISCLAIMER}\n")

    universe = fetch_universe_full_a()
    panel = load_panel(args.datadir, universe, args.end)
    feat = build_features(panel, sorted(GRID["n_days"]), args.exit_ma)

    outdir = Path(args.outdir)
    result = run_walkforward(panel, feat, folds, grid, args.exit_ma, outdir)

    # ---------- 稳健参数与默认参数的全样本期评估 ----------
    robust_stats, robust_curve = evaluate_full(
        panel, feat, result["robust_combo"], args.exit_ma, args.start, args.end)
    default_stats, default_curve = evaluate_full(
        panel, feat, result["default_combo"], args.exit_ma, args.start, args.end)
    full = pd.DataFrame([robust_stats, default_stats])
    full.to_csv(outdir / "robust_fullperiod.csv", index=False, encoding="utf-8-sig")
    robust_curve.to_csv(outdir / "pool_mean_equity_robust.csv",
                        index=False, encoding="utf-8-sig")
    default_curve.to_csv(outdir / "pool_mean_equity_default.csv",
                         index=False, encoding="utf-8-sig")
    print(f"\n[ok] 稳健参数与默认参数全样本期对比已保存 robust_fullperiod.csv")
    print(pd.DataFrame({
        "稳健参数": {k: full.iloc[0][k] for k in
                    ("params", "pool_total", "pool_annual", "pool_mdd",
                     "pool_win_rate", "pool_n_traded")},
        "TSP默认参数": {k: full.iloc[1][k] for k in
                      ("params", "pool_total", "pool_annual", "pool_mdd",
                       "pool_win_rate", "pool_n_traded")},
    }).to_string())

    if not args.no_plot:
        plot_heatmap(result["grid_df"], result["robust_combo"],
                     result["is_best_combo"], outdir / "heatmap_pool.png")
        plot_scatter(result["grid_df"], result["robust_combo"],
                     result["is_best_combo"], result["default_combo"],
                     result["rho"], result["is_best_mean_oos_pct"],
                     result["flip_rate"], outdir / "scatter_pool.png")
        plot_curves_compare(robust_curve, default_curve,
                            outdir / "robust_compare.png")

    print(f"\n⚠ {DISCLAIMER}")
    return 0


def plot_curves_compare(robust_curve: pd.DataFrame, default_curve: pd.DataFrame,
                        out_png: Path) -> None:
    """稳健参数 vs 默认参数的全池平均净值曲线对比图。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(pd.to_datetime(robust_curve["date"]), robust_curve["mean_strat"],
            label="稳健参数（全池等权平均净值）", lw=1.5, color="tab:blue")
    ax.plot(pd.to_datetime(default_curve["date"]), default_curve["mean_strat"],
            label="TSP 默认参数（全池等权平均净值）", lw=1.5, color="tab:orange")
    ax.set_title("n_day_low_reversal 全池等权平均净值：稳健参数 vs TSP 默认参数"
                 "（小数股全仓口径）")
    ax.set_ylabel("净值（窗口起点=1）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 净值对比图已保存: {out_png}")


if __name__ == "__main__":
    sys.exit(main())
