"""开源经典因子库移植：WorldQuant Alpha101 + Qlib Alpha158（量价部分）。

任务来源：GitHub issue #36（"扩大因子库"）。本模块把两个被广泛使用的开源
因子库移植到本项目的数据口径（A 股 qfq 日线，OHLCV + 成交额）：

1. **Alpha101**（Kakushadze 2016, "101 Formulaic Alphas"）：101 个公式化
   量价 alpha。其中 82 个只依赖 OHLCV（含 vwap/returns 派生量）；18 个
   依赖行业分类（IndNeutralize/indneutralize），1 个（alpha056）依赖
   市值 cap——这两类在调用方不提供对应输入时返回 NaN（见
   ALPHA101_NEEDS_INDUSTRY / ALPHA101_NEEDS_CAP）。

2. **Alpha158 量价部分**（Microsoft Qlib，qlib/contrib/data/handler.py 的
   Alpha158 默认配置）：9 个 K 线特征（KMID/KLEN/KMID2/KUP/KUP2/KLOW/
   KLOW2/KSFT/KSFT2）+ 4 个原始价格特征（OPEN0/HIGH0/LOW0/VWAP0）+
   29 种滚动算子 × 5 窗口（5/10/20/30/60），共 158 个特征。Qlib 语义
   **逐条对齐**：滚动算子 min_periods=1（不足窗口用已有值）、Std 用
   ddof=1、Rank 用 pandas rolling rank(pct=True)、Slope/Rsquare/Resi 用
   qlib 的窗口内 OLS 定义（x 取窗口位置，最新值 x=窗口长度）、Corr 在
   任一侧滚动 std(ddof=1)≈0（atol 2e-5）时置 NaN、IdxMax/IdxMin 返回
   窗口内最大值/最小值的位置（1 起）。

口径约定（与本项目其余部分一致，详见 docs/optimization-log.md 第 18 章）：
- **无未来函数**：全部时间序列算子只使用当日及之前的信息（shift/rolling，
  窗口右端 = 当日）。selftest 用"截断一致性 + 未来扰动不变性 + 停牌缺口"
  三组断言守护。
- **vwap 近似**：vwap = 成交额 / 成交量（本项目日线无真实 VWAP 字段，
  与 qlib/WorldQuant 数据约定不同，如实声明）。
- **returns** = close.pct_change()（当日相对前一日收盘的涨跌幅）。
- **rank(x)**：按论文原义实现为**横截面**分位秩（同一日期、全股票池内
  的 pct rank，axis=1）。注意：公开的单股票移植版（如 yli188 的 pandas
  版）因数据形态限制把 rank 退化为时间序列秩，本实现恢复论文语义。
- **scale(x)**：横截面归一化 x / Σ|x|（Σ 取同一日期的全股票池）。
- **小数窗口**：论文里 3.92795 之类的窗口按四舍五入取整（与公开移植版
  一致），原值保留在注释中。
- **IndNeutralize / indneutralize**：x 减去同行业组内均值（横截面）。
  论文区分 sector/industry/subindustry 三级分类；本项目尚无该数据，
  统一接受调用方提供的一个行业标签序列（如申万一级）。
- **数据卫生**：宽表为并集交易日历（日期 × 股票），停牌/未上市日为 NaN。
  时间序列算子按**个股自身连续交易日**计算（压缩掉 NaN 后滑动窗口，与
  第 6 步 factor_volatility 同口径），结果放回并集日历原位（停牌日 NaN）；
  横截面算子（rank/scale/indneutralize）在同一日期对非 NaN 股票计算。
  除零/对非正数取对数等产生的 ±inf 在结果中统一替换为 NaN。

来源：
- Kakushadze (2016), "101 Formulaic Alphas", Wilmott —
  https://arxiv.org/abs/1601.00991
- Microsoft Qlib（MIT License）Alpha158 定义：
  https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
  https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- 公开参考实现（公式对照，未直接引用代码）：
  https://github.com/yli188/WorldQuant_alpha101_code

用法：
    python scripts/alpha_factors.py selftest   # 合成数据自检（不访问网络）
    python scripts/alpha_factors.py info       # 打印因子清单与数据要求
    # 对本地 data/daily 下的几只股票计算并输出最新一期横截面
    python scripts/alpha_factors.py compute --symbols 000001 600519 600036 300750
    # 全 A 池：先经 screener.py 的 TSP parquet 导入，再对前 N 只计算
    python scripts/alpha_factors.py compute --pool full_a --source tsp --max-stocks 400
    # 复用已有导入缓存（--no-fetch 跳过拉取/导入，直接用 --datadir 下的 CSV）
    python scripts/alpha_factors.py compute --no-fetch --max-stocks 400 --datadir data/daily

输出（output/factors/）：
    alpha101_latest.csv            每只股票最新一期的 101 个 alpha 值（股票 × 因子）
    alpha158_latest.csv            每只股票最新一期的 158 个 Alpha158 特征
    alpha101_coverage_summary.csv  每个 alpha 的有效值占比与描述统计（因子 × 指标）
    alpha158_coverage_summary.csv  每个 Alpha158 特征的有效值占比与描述统计

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/alpha_factors.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import DISCLAIMER, standardize_stock  # noqa: E402

# ---------------------------------------------------------------------------
# 常量与清单
# ---------------------------------------------------------------------------

# Alpha158 滚动算子窗口（Qlib 默认）
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]

# Alpha158 K 线特征（Qlib Alpha158DL "kbar" 配置）
ALPHA158_KBAR = ["KMID", "KLEN", "KMID2", "KUP", "KUP2",
                 "KLOW", "KLOW2", "KSFT", "KSFT2"]

# Alpha158 原始价格特征（Qlib handler.py 默认配置：windows=[0]，
# feature=[OPEN, HIGH, LOW, VWAP]，均除以 $close 去单位）
ALPHA158_PRICE0 = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]

# Alpha158 滚动算子（Qlib Alpha158DL "rolling" 默认配置：29 种 × 5 窗口）
ALPHA158_ROLLING = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN",
    "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
    "CORR", "CORD", "CNTP", "CNTN", "CNTD",
    "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA",
    "VSUMP", "VSUMN", "VSUMD",
]

ALPHA158_NAMES = (
    ALPHA158_KBAR
    + ALPHA158_PRICE0
    + [f"{op}{w}" for w in ALPHA158_WINDOWS for op in ALPHA158_ROLLING]
)
assert len(ALPHA158_NAMES) == 158, len(ALPHA158_NAMES)

ALPHA101_NAMES = [f"alpha{i:03d}" for i in range(1, 102)]

# 依赖行业分类（IndNeutralize/indneutralize）的 18 个 alpha（论文原文）
ALPHA101_NEEDS_INDUSTRY = {
    "alpha048", "alpha058", "alpha059", "alpha063", "alpha067",
    "alpha069", "alpha070", "alpha076", "alpha079", "alpha080",
    "alpha082", "alpha087", "alpha089", "alpha090", "alpha091",
    "alpha093", "alpha097", "alpha100",
}
# 依赖市值 cap 的 alpha（论文原文 rank(returns * cap)）
ALPHA101_NEEDS_CAP = {"alpha056"}
# 只用 OHLCV(+vwap/returns 派生量) 的 82 个 alpha
ALPHA101_PRICE_VOLUME = [
    n for n in ALPHA101_NAMES
    if n not in ALPHA101_NEEDS_INDUSTRY and n not in ALPHA101_NEEDS_CAP
]
assert len(ALPHA101_PRICE_VOLUME) == 82, len(ALPHA101_PRICE_VOLUME)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _nan_like(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)


def _map_cols(frame: pd.DataFrame, fn) -> pd.DataFrame:
    """对宽表（并集日历 × 股票）逐列应用 fn(1-D ndarray) -> 1-D ndarray。

    口径（与项目第 6/8 步一致）：每列先压缩掉 NaN 行——上市前与**停牌日**
    （个股 CSV 无该日行情，pivot 后为 NaN）不参与计算，窗口算子只在个股
    自身的连续交易日上滑动；结果放回并集日历原位（停牌日保持 NaN）。
    派生序列内部由热身混入的 NaN 同样被跳过，不影响其后窗口的结果。
    """
    cols = {}
    for col in frame.columns:
        s = frame[col].to_numpy(dtype=float)
        valid = ~np.isnan(s)
        out = np.full(len(s), np.nan)
        if valid.any():
            out[valid] = fn(s[valid])
        cols[col] = out
    return pd.DataFrame(cols, index=frame.index)


def _map_pair(a: pd.DataFrame, b: pd.DataFrame, fn) -> pd.DataFrame:
    """同 _map_cols，但 fn 接受两个 1-D ndarray（逐列配对，按共同有效行压缩）。"""
    cols = {}
    for col in a.columns:
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        valid = ~np.isnan(x) & ~np.isnan(y)
        out = np.full(len(x), np.nan)
        if valid.any():
            out[valid] = fn(x[valid], y[valid])
        cols[col] = out
    return pd.DataFrame(cols, index=a.index)


def _clean_inf(frame: pd.DataFrame) -> pd.DataFrame:
    """±inf 统一替换为 NaN（除零等口径见模块 docstring）。"""
    return frame.replace([np.inf, -np.inf], np.nan)


def _as_frame(x, like: pd.DataFrame) -> pd.DataFrame:
    """把标量/ndarray 包装成与 like 同 index/columns 的 DataFrame。"""
    if isinstance(x, pd.DataFrame):
        return x
    if np.isscalar(x):
        return pd.DataFrame(np.full((like.shape[0], like.shape[1]), float(x)),
                            index=like.index, columns=like.columns)
    return pd.DataFrame(np.asarray(x, dtype=float),
                        index=like.index, columns=like.columns)


# ---------------------------------------------------------------------------
# Alpha101 时间序列算子（WorldQuant 口径：完整窗口，热身期 NaN）
# ---------------------------------------------------------------------------

def _shift_np(a: np.ndarray, d: int) -> np.ndarray:
    """个股自身交易日的滞后（跳过停牌日，与并集日历 shift 不同）。"""
    n = len(a)
    if d >= n:
        return np.full(n, np.nan)
    return np.concatenate([np.full(d, np.nan), a[:-d]])


def _pct_change_np(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / _shift_np(a, 1)


def _delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _map_cols(x, lambda a: _shift_np(a, d))


def _delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    with np.errstate(invalid="ignore"):
        return _map_cols(x, lambda a: a - _shift_np(a, d))


def _rolling_np(fn):
    """把一个 pandas Series 滚动函数包装为逐列 ndarray 函数。"""
    def inner(a: np.ndarray, w: int) -> np.ndarray:
        return fn(pd.Series(a), w)
    return inner


def _ts_mean(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return _map_cols(x, lambda a: pd.Series(a).rolling(w, min_periods=w)
                     .mean().to_numpy())


def _ts_std(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """stddev：pandas rolling std（ddof=1，与常用 pandas 移植版一致）。"""
    return _map_cols(x, lambda a: pd.Series(a).rolling(w, min_periods=w)
                     .std().to_numpy())


def _ts_sum(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return _map_cols(x, lambda a: pd.Series(a).rolling(w, min_periods=w)
                     .sum().to_numpy())


def _ts_min(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return _map_cols(x, lambda a: pd.Series(a).rolling(w, min_periods=w)
                     .min().to_numpy())


def _ts_max(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return _map_cols(x, lambda a: pd.Series(a).rolling(w, min_periods=w)
                     .max().to_numpy())


def _ts_rank(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """窗口内最后一值的秩（scipy rankdata ties='average' 口径，1 起）。

    含 NaN 的窗口返回 NaN（与 pandas rolling 语义一致）。
    """
    def fn(a: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        last = win[:, -1:]
        vals = (win < last).sum(axis=1) + ((win == last).sum(axis=1) + 1) / 2.0
        vals[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1:] = vals
        return out
    return _map_cols(x, fn)


def _ts_argmax(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """窗口内最大值的位置（1 起，平局取最先出现）。含 NaN 的窗口返回 NaN。"""
    def fn(a: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        vals = np.argmax(win, axis=1).astype(float) + 1
        vals[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1:] = vals
        return out
    return _map_cols(x, fn)


def _ts_argmin(x: pd.DataFrame, w: int) -> pd.DataFrame:
    def fn(a: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        vals = np.argmin(win, axis=1).astype(float) + 1
        vals[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1:] = vals
        return out
    return _map_cols(x, fn)


def _decay_linear(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """线性衰减加权平均，权重 1..w（最新权重最大）。含 NaN 的窗口返回 NaN。"""
    weights = np.arange(1, w + 1, dtype=float) / (w * (w + 1) / 2.0)

    def fn(a: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        with np.errstate(invalid="ignore"):
            vals = win @ weights
        vals[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1:] = vals
        return out
    return _map_cols(x, fn)


def _product(x: pd.DataFrame, w: int) -> pd.DataFrame:
    def fn(a: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        with np.errstate(invalid="ignore"):
            vals = win.prod(axis=1)
        vals[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1:] = vals
        return out
    return _map_cols(x, fn)


def _corr(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动相关系数（完整窗口，pandas rolling corr 同口径）。"""
    def fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        def rsum(z: np.ndarray) -> np.ndarray:
            cs = np.cumsum(z)
            return cs[w - 1:] - np.concatenate([[0.0], cs[:-w]])
        sa, sb = rsum(a), rsum(b)
        sab, sa2, sb2 = rsum(a * b), rsum(a * a), rsum(b * b)
        num = w * sab - sa * sb
        with np.errstate(divide="ignore", invalid="ignore"):
            den = np.sqrt(np.maximum((w * sa2 - sa ** 2) * (w * sb2 - sb ** 2), 0.0))
            out[w - 1:] = num / den
        return out
    return _map_pair(x, y, fn)


def _cov(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动协方差（pandas rolling cov 口径：ddof=1）。"""
    def fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        out = np.full(n, np.nan)
        if n < w:
            return out
        def rsum(z: np.ndarray) -> np.ndarray:
            cs = np.cumsum(z)
            return cs[w - 1:] - np.concatenate([[0.0], cs[:-w]])
        sa, sb = rsum(a), rsum(b)
        sab = rsum(a * b)
        out[w - 1:] = (w * sab - sa * sb) / (w * (w - 1))
        return out
    return _map_pair(x, y, fn)


# ---------------------------------------------------------------------------
# Alpha101 横截面算子（WorldQuant 口径：同一日期、全股票池）
# ---------------------------------------------------------------------------

def _rank_cs(x: pd.DataFrame) -> pd.DataFrame:
    """rank(x)：横截面分位秩（axis=1，NaN 保持 NaN）。"""
    return x.rank(axis=1, pct=True)


def _scale(x: pd.DataFrame) -> pd.DataFrame:
    """scale(x)：x / Σ|x|（Σ 取同一日期的全股票池，NaN 不参与）。"""
    denom = x.abs().sum(axis=1).replace(0.0, np.nan)
    return x.div(denom, axis=0)


def _indneutralize(x: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """indneutralize(x, g)：x 减去同行业组内横截面均值。

    industry 为 symbol → 标签 的 Series；论文的 sector/industry/subindustry
    三级分类在本仓库统一由调用方提供的单一分类近似（见模块 docstring）。
    """
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for label, members in industry.groupby(industry).groups.items():
        cols = [c for c in members if c in x.columns]
        if not cols:
            continue
        mean = x[cols].mean(axis=1)
        for c in cols:
            out[c] = x[c] - mean
    return out


def _tern(cond: pd.DataFrame, a, b) -> pd.DataFrame:
    """三元选择：cond 为真取 a，否则取 b；cond 为 NaN 处结果保持 NaN。

    a/b 可为 DataFrame 或标量（与论文的 ?: 语义一致，热身期不产生伪值）。
    """
    like = cond
    a_f = _as_frame(a, like).to_numpy(dtype=float)
    b_f = _as_frame(b, like).to_numpy(dtype=float)
    out = np.where(cond.fillna(False).to_numpy(dtype=bool), a_f, b_f)
    return pd.DataFrame(out, index=like.index, columns=like.columns).mask(
        cond.isna())


def _lt(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """逐元素小于比较，结果 0/1；任一侧为 NaN（停牌等）时结果 NaN，
    避免把停牌日误计为"条件不成立"。"""
    return (x < y).astype(float).mask(x.isna() | y.isna())


# 标量运算
def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x)


def _signedpow(x: pd.DataFrame, a) -> pd.DataFrame:
    """signedpower(x, a) = sign(x) * |x|^a（a 可为标量或同形状 DataFrame）。"""
    a_f = _as_frame(a, x)
    return np.sign(x) * x.abs().pow(a_f)


def _max2(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.maximum(x.to_numpy(), y.to_numpy()),
                        index=x.index, columns=x.columns)


def _min2(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.minimum(x.to_numpy(), y.to_numpy()),
                        index=x.index, columns=x.columns)


# ---------------------------------------------------------------------------
# 计算上下文：基础序列 + 惰性缓存
# ---------------------------------------------------------------------------

class _Ctx:
    """Alpha101 计算上下文。

    frames: open/high/low/close/volume/vwap/returns 宽表（日期 × 股票）；
    cap: 可选，symbol → 市值 Series（alpha056 需要，否则该 alpha 全 NaN）；
    industry: 可选，symbol → 行业标签 Series（18 个行业 alpha 需要）。
    memo 缓存滚动算子的结果，避免同一子表达式重复计算。
    """

    def __init__(self, frames: dict[str, pd.DataFrame],
                 cap: pd.Series | None = None,
                 industry: pd.Series | None = None) -> None:
        self.frames = frames
        self.cap = cap
        self.industry = industry
        self.memo: dict = {}
        self._warned: set[str] = set()

    @property
    def o(self) -> pd.DataFrame:
        return self.frames["open"]

    @property
    def h(self) -> pd.DataFrame:
        return self.frames["high"]

    @property
    def l(self) -> pd.DataFrame:
        return self.frames["low"]

    @property
    def c(self) -> pd.DataFrame:
        return self.frames["close"]

    @property
    def v(self) -> pd.DataFrame:
        return self.frames["volume"]

    @property
    def vwap(self) -> pd.DataFrame:
        return self.frames["vwap"]

    @property
    def returns(self) -> pd.DataFrame:
        return self.frames["returns"]

    def get(self, key, fn):
        """按 key 惰性缓存公共子表达式（如某个滚动算子/横截面秩）。"""
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]

    def adv(self, n: int) -> pd.DataFrame:
        """advN：N 日平均成交量（论文的 AverageDollarVolume，本项目用
        手数成交量近似，见第 18 章口径说明）。"""
        return self.get(("adv", n), lambda: _ts_mean(self.v, n))

    def rank_of(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.get(("rank", id(frame)), lambda: _rank_cs(frame))

    def rank(self, field: str) -> pd.DataFrame:
        mapping = {"open": self.o, "high": self.h, "low": self.l,
                   "close": self.c, "volume": self.v, "vwap": self.vwap,
                   "returns": self.returns}
        return self.rank_of(mapping[field])

    def warn_missing(self, name: str, what: str) -> None:
        if name not in self._warned:
            print(f"[warn] {name} 需要 {what}，未提供输入 → 结果全为 NaN",
                  file=sys.stderr)
            self._warned.add(name)


# ---------------------------------------------------------------------------
# Alpha101：101 个公式（论文原文保留在注释中，小数窗口已按四舍五入取整）
# ---------------------------------------------------------------------------

def _a001(ctx: _Ctx) -> pd.DataFrame:
    # rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    inner = ctx.c.where(ctx.returns < 0, ctx.get(("st", "returns", 20),
                                                 lambda: _ts_std(ctx.returns, 20)))
    return _rank_cs(_ts_argmax(_signedpow(inner, 2.0), 5)) - 0.5


def _a002(ctx: _Ctx) -> pd.DataFrame:
    # -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
    return -_corr(ctx.rank_of(_delta(_log(ctx.v), 2)),
                  ctx.rank_of((ctx.c - ctx.o) / ctx.o), 6)


def _a003(ctx: _Ctx) -> pd.DataFrame:
    # -1 * correlation(rank(open), rank(volume), 10)
    return -_corr(ctx.rank("open"), ctx.rank("volume"), 10)


def _a004(ctx: _Ctx) -> pd.DataFrame:
    # -1 * Ts_Rank(rank(low), 9)
    return -_ts_rank(ctx.rank("low"), 9)


def _a005(ctx: _Ctx) -> pd.DataFrame:
    # rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
    return (_rank_cs(ctx.o - _ts_sum(ctx.vwap, 10) / 10)
            * -_abs(_rank_cs(ctx.c - ctx.vwap)))


def _a006(ctx: _Ctx) -> pd.DataFrame:
    # -1 * correlation(open, volume, 10)
    return -_corr(ctx.o, ctx.v, 10)


def _a007(ctx: _Ctx) -> pd.DataFrame:
    # ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    val = -_ts_rank(_abs(_delta(ctx.c, 7)), 60) * _sign(_delta(ctx.c, 7))
    return _tern(ctx.adv(20) < ctx.v, val, -1.0)


def _a008(ctx: _Ctx) -> pd.DataFrame:
    # -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10)))
    x = _ts_sum(ctx.o, 5) * _ts_sum(ctx.returns, 5)
    return -_rank_cs(x - _delay(x, 10))


def _a009(ctx: _Ctx) -> pd.DataFrame:
    # ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
    #  ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    dc = _delta(ctx.c, 1)
    cond = (_ts_min(dc, 5) > 0) | (_ts_max(dc, 5) < 0)
    return _tern(cond, dc, -dc)


def _a010(ctx: _Ctx) -> pd.DataFrame:
    # rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
    #       ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    dc = _delta(ctx.c, 1)
    cond = (_ts_min(dc, 4) > 0) | (_ts_max(dc, 4) < 0)
    return _rank_cs(_tern(cond, dc, -dc))


def _a011(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    d = ctx.vwap - ctx.c
    return ((_rank_cs(_ts_max(d, 3)) + _rank_cs(_ts_min(d, 3)))
            * _rank_cs(_delta(ctx.v, 3)))


def _a012(ctx: _Ctx) -> pd.DataFrame:
    # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    return _sign(_delta(ctx.v, 1)) * -_delta(ctx.c, 1)


def _a013(ctx: _Ctx) -> pd.DataFrame:
    # -1 * rank(covariance(rank(close), rank(volume), 5))
    return -_rank_cs(_cov(ctx.rank("close"), ctx.rank("volume"), 5))


def _a014(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    return -_rank_cs(_delta(ctx.returns, 3)) * _corr(ctx.o, ctx.v, 10)


def _a015(ctx: _Ctx) -> pd.DataFrame:
    # -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    return -_ts_sum(_rank_cs(_corr(ctx.rank("high"), ctx.rank("volume"), 3)), 3)


def _a016(ctx: _Ctx) -> pd.DataFrame:
    # -1 * rank(covariance(rank(high), rank(volume), 5))
    return -_rank_cs(_cov(ctx.rank("high"), ctx.rank("volume"), 5))


def _a017(ctx: _Ctx) -> pd.DataFrame:
    # (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
    #  * rank(ts_rank((volume / adv20), 5)))
    return (-_rank_cs(_ts_rank(ctx.c, 10))
            * _rank_cs(_delta(_delta(ctx.c, 1), 1))
            * _rank_cs(_ts_rank(ctx.v / ctx.adv(20), 5)))


def _a018(ctx: _Ctx) -> pd.DataFrame:
    # -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10)))
    return -_rank_cs(_ts_std(_abs(ctx.c - ctx.o), 5) + (ctx.c - ctx.o)
                     + _corr(ctx.c, ctx.o, 10))


def _a019(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    return (-_sign((ctx.c - _delay(ctx.c, 7)) + _delta(ctx.c, 7))
            * (1 + _rank_cs(1 + _ts_sum(ctx.returns, 250))))


def _a020(ctx: _Ctx) -> pd.DataFrame:
    # (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1))))
    #  * rank((open - delay(low, 1))))
    return (-_rank_cs(ctx.o - _delay(ctx.h, 1))
            * _rank_cs(ctx.o - _delay(ctx.c, 1))
            * _rank_cs(ctx.o - _delay(ctx.l, 1)))


def _a021(ctx: _Ctx) -> pd.DataFrame:
    # (((sum(close, 8) / 8 + stddev(close, 8)) < (sum(close, 2) / 2)) ? -1 :
    #  (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 :
    #   (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : -1)))
    sma8 = _ts_mean(ctx.c, 8)
    sd8 = _ts_std(ctx.c, 8)
    sma2 = _ts_mean(ctx.c, 2)
    vol_ratio = ctx.v / ctx.adv(20)
    cond_a = sma8 + sd8 < sma2
    cond_b = sma2 < sma8 - sd8
    cond_c = vol_ratio >= 1
    ready = ~(cond_a.isna() | cond_b.isna() | cond_c.isna())
    out = np.where(cond_a.fillna(False), -1.0,
                   np.where(cond_b.fillna(False) | cond_c.fillna(False), 1.0, -1.0))
    return pd.DataFrame(out, index=ctx.c.index,
                        columns=ctx.c.columns).mask(~ready)


def _a022(ctx: _Ctx) -> pd.DataFrame:
    # -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
    return -_delta(_corr(ctx.h, ctx.v, 5), 5) * _rank_cs(_ts_std(ctx.c, 20))


def _a023(ctx: _Ctx) -> pd.DataFrame:
    # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    return _tern(_ts_mean(ctx.h, 20) < ctx.h, -_delta(ctx.h, 2), 0.0)


def _a024(ctx: _Ctx) -> pd.DataFrame:
    # ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05)
    #   || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05))
    #  ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    inner = _delta(_ts_mean(ctx.c, 100), 100) / _delay(ctx.c, 100)
    return _tern(inner <= 0.05, -(ctx.c - _ts_min(ctx.c, 100)),
                 -_delta(ctx.c, 3))


def _a025(ctx: _Ctx) -> pd.DataFrame:
    # rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    return _rank_cs(((-ctx.returns) * ctx.adv(20)) * ctx.vwap * (ctx.h - ctx.c))


def _a026(ctx: _Ctx) -> pd.DataFrame:
    # -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    return -_ts_max(_corr(_ts_rank(ctx.v, 5), _ts_rank(ctx.h, 5), 5), 3)


def _a027(ctx: _Ctx) -> pd.DataFrame:
    # ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    x = _rank_cs(_ts_sum(_corr(ctx.rank("volume"), ctx.rank("vwap"), 6), 2) / 2.0)
    return _tern(x > 0.5, -1.0, 1.0)


def _a028(ctx: _Ctx) -> pd.DataFrame:
    # scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    return _scale(_corr(ctx.adv(20), ctx.l, 5) + (ctx.h + ctx.l) / 2 - ctx.c)


def _a029(ctx: _Ctx) -> pd.DataFrame:
    # (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5)
    #  + ts_rank(delay((-1 * returns), 6), 5))
    # 等价化简：sum(·,1)/product(·,1) 为单位元；delta(close-1, 5) = delta(close, 5)
    x = _ts_min(_rank_cs(_rank_cs(-_rank_cs(_delta(ctx.c, 5)))), 2)
    x = _rank_cs(_rank_cs(_scale(_log(x))))
    return _ts_min(x, 5) + _ts_rank(_delay(-ctx.returns, 6), 5)


def _a030(ctx: _Ctx) -> pd.DataFrame:
    # (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))))
    #   + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    dc = _delta(ctx.c, 1)
    inner = _sign(dc) + _sign(_delay(dc, 1)) + _sign(_delay(dc, 2))
    return (1.0 - _rank_cs(inner)) * _ts_sum(ctx.v, 5) / _ts_sum(ctx.v, 20)


def _a031(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
    #   + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    p1 = _rank_cs(_rank_cs(_rank_cs(
        _decay_linear(-_rank_cs(_rank_cs(_delta(ctx.c, 10))), 10))))
    p2 = _rank_cs(-_delta(ctx.c, 3))
    p3 = _sign(_scale(_corr(ctx.adv(20), ctx.l, 12)))
    return p1 + p2 + p3


def _a032(ctx: _Ctx) -> pd.DataFrame:
    # (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    return (_scale(_ts_sum(ctx.c, 7) / 7 - ctx.c)
            + 20 * _scale(_corr(ctx.vwap, _delay(ctx.c, 5), 230)))


def _a033(ctx: _Ctx) -> pd.DataFrame:
    # rank((-1 * ((1 - (open / close))^1)))
    return _rank_cs(-(1 - ctx.o / ctx.c))


def _a034(ctx: _Ctx) -> pd.DataFrame:
    # rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    inner = _ts_std(ctx.returns, 2) / _ts_std(ctx.returns, 5)
    return _rank_cs(2 - _rank_cs(inner) - _rank_cs(_delta(ctx.c, 1)))


def _a035(ctx: _Ctx) -> pd.DataFrame:
    # ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    return (_ts_rank(ctx.v, 32)
            * (1 - _ts_rank(ctx.c + ctx.h - ctx.l, 16))
            * (1 - _ts_rank(ctx.returns, 32)))


def _a036(ctx: _Ctx) -> pd.DataFrame:
    # (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close))))
    #    + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6))))
    #   + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    return (2.21 * _rank_cs(_corr(ctx.c - ctx.o, _delay(ctx.v, 1), 15))
            + 0.7 * _rank_cs(ctx.o - ctx.c)
            + 0.73 * _rank_cs(_ts_rank(_delay(-ctx.returns, 6), 5))
            + _rank_cs(_abs(_corr(ctx.vwap, ctx.adv(20), 6)))
            + 0.6 * _rank_cs((_ts_mean(ctx.c, 200) - ctx.o) * (ctx.c - ctx.o)))


def _a037(ctx: _Ctx) -> pd.DataFrame:
    # (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    return (_rank_cs(_corr(_delay(ctx.o - ctx.c, 1), ctx.c, 200))
            + _rank_cs(ctx.o - ctx.c))


def _a038(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    return -_rank_cs(_ts_rank(ctx.c, 10)) * _rank_cs(ctx.c / ctx.o)


def _a039(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9))))))
    #  * (1 + rank(sum(returns, 250))))
    return (-_rank_cs(_delta(ctx.c, 7)
                      * (1 - _rank_cs(_decay_linear(ctx.v / ctx.adv(20), 9))))
            * (1 + _rank_cs(_ts_sum(ctx.returns, 250))))


def _a040(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    return -_rank_cs(_ts_std(ctx.h, 10)) * _corr(ctx.h, ctx.v, 10)


def _a041(ctx: _Ctx) -> pd.DataFrame:
    # (((high * low)^0.5) - vwap)
    return (ctx.h * ctx.l) ** 0.5 - ctx.vwap


def _a042(ctx: _Ctx) -> pd.DataFrame:
    # (rank((vwap - close)) / rank((vwap + close)))
    return _rank_cs(ctx.vwap - ctx.c) / _rank_cs(ctx.vwap + ctx.c)


def _a043(ctx: _Ctx) -> pd.DataFrame:
    # (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    return _ts_rank(ctx.v / ctx.adv(20), 20) * _ts_rank(-_delta(ctx.c, 7), 8)


def _a044(ctx: _Ctx) -> pd.DataFrame:
    # (-1 * correlation(high, rank(volume), 5))
    return -_corr(ctx.h, ctx.rank("volume"), 5)


def _a045(ctx: _Ctx) -> pd.DataFrame:
    # (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
    #        * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    return -(_rank_cs(_ts_mean(_delay(ctx.c, 5), 20))
             * _corr(ctx.c, ctx.v, 2)
             * _rank_cs(_corr(_ts_sum(ctx.c, 5), _ts_sum(ctx.c, 20), 2)))


def _a046(ctx: _Ctx) -> pd.DataFrame:
    # ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? -1 :
    #  (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
    #   ((-1 * 1) * (close - delay(close, 1)))))
    inner = (_delay(ctx.c, 20) - _delay(ctx.c, 10)) / 10 \
        - (_delay(ctx.c, 10) - ctx.c) / 10
    return _tern(inner > 0.25, -1.0,
                 _tern(inner < 0, 1.0, -(ctx.c - _delay(ctx.c, 1))))


def _a047(ctx: _Ctx) -> pd.DataFrame:
    # ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5)))
    #  - rank((vwap - delay(vwap, 5))))
    return ((_rank_cs(1 / ctx.c) * ctx.v) / ctx.adv(20)
            * (ctx.h * _rank_cs(ctx.h - ctx.c) / (_ts_sum(ctx.h, 5) / 5))
            - _rank_cs(ctx.vwap - _delay(ctx.vwap, 5)))


def _a048(ctx: _Ctx) -> pd.DataFrame:
    # (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close),
    #                IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    if ctx.industry is None:
        ctx.warn_missing("alpha048", "行业分类（industry）")
        return _nan_like(ctx.c)
    dc1 = _delta(ctx.c, 1)
    x = _corr(dc1, _delta(_delay(ctx.c, 1), 1), 250) * dc1 / ctx.c
    return _indneutralize(x, ctx.industry) / _ts_sum((dc1 / _delay(ctx.c, 1)) ** 2, 250)


def _a049(ctx: _Ctx) -> pd.DataFrame:
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1
    #  : ((-1 * 1) * (close - delay(close, 1))))
    inner = (_delay(ctx.c, 20) - _delay(ctx.c, 10)) / 10 \
        - (_delay(ctx.c, 10) - ctx.c) / 10
    return _tern(inner < -0.1, 1.0, -(ctx.c - _delay(ctx.c, 1)))


def _a050(ctx: _Ctx) -> pd.DataFrame:
    # (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    return -_ts_max(_rank_cs(_corr(ctx.rank("volume"), ctx.rank("vwap"), 5)), 5)


def _a051(ctx: _Ctx) -> pd.DataFrame:
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1
    #  : ((-1 * 1) * (close - delay(close, 1))))
    inner = (_delay(ctx.c, 20) - _delay(ctx.c, 10)) / 10 \
        - (_delay(ctx.c, 10) - ctx.c) / 10
    return _tern(inner < -0.05, 1.0, -(ctx.c - _delay(ctx.c, 1)))


def _a052(ctx: _Ctx) -> pd.DataFrame:
    # ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220)))
    #  * ts_rank(volume, 5))
    ml5 = _ts_min(ctx.l, 5)
    return ((-ml5 + _delay(ml5, 5))
            * _rank_cs((_ts_sum(ctx.returns, 240) - _ts_sum(ctx.returns, 20)) / 220)
            * _ts_rank(ctx.v, 5))


def _a053(ctx: _Ctx) -> pd.DataFrame:
    # -1 * delta((((close - low) - (high - close)) / (close - low)), 9)
    inner = ((ctx.c - ctx.l) - (ctx.h - ctx.c)) / (ctx.c - ctx.l)
    return -_delta(inner, 9)


def _a054(ctx: _Ctx) -> pd.DataFrame:
    # ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    return -(ctx.l - ctx.c) * ctx.o ** 5 / ((ctx.l - ctx.h) * ctx.c ** 5)


def _a055(ctx: _Ctx) -> pd.DataFrame:
    # -1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6)
    denom = _ts_max(ctx.h, 12) - _ts_min(ctx.l, 12)
    inner = (ctx.c - _ts_min(ctx.l, 12)) / denom
    return -_corr(_rank_cs(inner), ctx.rank("volume"), 6)


def _a056(ctx: _Ctx) -> pd.DataFrame:
    # (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    if ctx.cap is None:
        ctx.warn_missing("alpha056", "市值（cap）")
        return _nan_like(ctx.c)
    inner = _ts_sum(ctx.returns, 10) / _ts_sum(_ts_sum(ctx.returns, 2), 3)
    return -(_rank_cs(inner) * _rank_cs(ctx.returns * ctx.cap))


def _a057(ctx: _Ctx) -> pd.DataFrame:
    # (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    return -((ctx.c - ctx.vwap)
             / _decay_linear(_rank_cs(_ts_argmax(ctx.c, 30)), 2))


def _a058(ctx: _Ctx) -> pd.DataFrame:
    # (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    if ctx.industry is None:
        ctx.warn_missing("alpha058", "行业分类（industry）")
        return _nan_like(ctx.c)
    return -_ts_rank(_decay_linear(_corr(_indneutralize(ctx.vwap, ctx.industry),
                                         ctx.v, 4), 8), 6)


def _a059(ctx: _Ctx) -> pd.DataFrame:
    # (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))),
    #     IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    # 注：vwap*0.728317 + vwap*(1-0.728317) = vwap（论文原式的自加权恒等式）
    if ctx.industry is None:
        ctx.warn_missing("alpha059", "行业分类（industry）")
        return _nan_like(ctx.c)
    return -_ts_rank(_decay_linear(_corr(_indneutralize(ctx.vwap, ctx.industry),
                                         ctx.v, 4), 16), 8)


def _a060(ctx: _Ctx) -> pd.DataFrame:
    # (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume))))
    #            - scale(rank(ts_argmax(close, 10))))))
    inner = ((ctx.c - ctx.l) - (ctx.h - ctx.c)) / (ctx.h - ctx.l) * ctx.v
    return -(2 * _scale(_rank_cs(inner)) - _scale(_rank_cs(_ts_argmax(ctx.c, 10))))


def _a061(ctx: _Ctx) -> pd.DataFrame:
    # (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    return _lt(_rank_cs(ctx.vwap - _ts_min(ctx.vwap, 16)),
               _rank_cs(_corr(ctx.vwap, ctx.adv(180), 18)))


def _a062(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
    #   rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    inner = (ctx.rank("open") + ctx.rank("open")) \
        < (ctx.rank_of((ctx.h + ctx.l) / 2) + ctx.rank("high"))
    return _lt(_rank_cs(_corr(ctx.vwap, _ts_sum(ctx.adv(20), 22), 10)),
               _rank_cs(inner)) * -1


def _a063(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))
    #   - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))),
    #       sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha063", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _rank_cs(_decay_linear(_delta(_indneutralize(ctx.c, ctx.industry), 2), 8))
    blend = ctx.vwap * 0.318108 + ctx.o * (1 - 0.318108)
    p2 = _rank_cs(_decay_linear(_corr(blend, _ts_sum(ctx.adv(180), 37), 14), 12))
    return (p1 - p2) * -1


def _a064(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
    #                     sum(adv120, 12.7054), 16.6208)) <
    #   rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    x = _ts_sum(ctx.o * 0.178404 + ctx.l * (1 - 0.178404), 13)
    y = _ts_sum(ctx.adv(120), 13)
    z = _delta((ctx.h + ctx.l) / 2 * 0.178404 + ctx.vwap * (1 - 0.178404), 4)
    return _lt(_rank_cs(_corr(x, y, 17)), _rank_cs(z)) * -1


def _a065(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) <
    #   rank((open - ts_min(open, 13.635)))) * -1)
    x = ctx.o * 0.00817205 + ctx.vwap * (1 - 0.00817205)
    return _lt(_rank_cs(_corr(x, _ts_sum(ctx.adv(60), 9), 6)),
               _rank_cs(ctx.o - _ts_min(ctx.o, 14))) * -1


def _a066(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
    #   Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap)
    #                         / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    p1 = _rank_cs(_decay_linear(_delta(ctx.vwap, 4), 7))
    p2 = _ts_rank(_decay_linear((ctx.l - ctx.vwap)
                                / (ctx.o - (ctx.h + ctx.l) / 2), 11), 7)
    return (p1 + p2) * -1


def _a067(ctx: _Ctx) -> pd.DataFrame:
    # ((rank((high - ts_min(high, 2.14593)))^
    #   rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha067", "行业分类（industry）")
        return _nan_like(ctx.c)
    base = _rank_cs(ctx.h - _ts_min(ctx.h, 2))
    exp = _rank_cs(_corr(_indneutralize(ctx.vwap, ctx.industry),
                         _indneutralize(ctx.adv(20), ctx.industry), 6))
    return _signedpow(base, exp) * -1


def _a068(ctx: _Ctx) -> pd.DataFrame:
    # ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <
    #   rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    x = _ts_rank(_corr(ctx.rank("high"), ctx.rank_of(ctx.adv(15)), 9), 14)
    y = _rank_cs(_delta(ctx.c * 0.518371 + ctx.l * (1 - 0.518371), 1))
    return _lt(x, y) * -1


def _a069(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^
    #   Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha069", "行业分类（industry）")
        return _nan_like(ctx.c)
    base = _rank_cs(_ts_max(_delta(_indneutralize(ctx.vwap, ctx.industry), 3), 5))
    exp = _ts_rank(_corr(ctx.c * 0.490655 + ctx.vwap * (1 - 0.490655),
                         ctx.adv(20), 5), 9)
    return _signedpow(base, exp) * -1


def _a070(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(delta(vwap, 1.29456))^
    #   Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha070", "行业分类（industry）")
        return _nan_like(ctx.c)
    base = _rank_cs(_delta(ctx.vwap, 1))
    exp = _ts_rank(_corr(_indneutralize(ctx.c, ctx.industry), ctx.adv(50), 18), 18)
    return _signedpow(base, exp) * -1


def _a071(ctx: _Ctx) -> pd.DataFrame:
    # (max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948),
    #      Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))
    p1 = _ts_rank(_decay_linear(_corr(_ts_rank(ctx.c, 3), _ts_rank(ctx.adv(180), 12), 18), 4), 16)
    p2 = _ts_rank(_decay_linear(_rank_cs(ctx.l + ctx.o - 2 * ctx.vwap) ** 2, 16), 4)
    return _max2(p1, p2)


def _a072(ctx: _Ctx) -> pd.DataFrame:
    # (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
    #  rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    p1 = _rank_cs(_decay_linear(_corr((ctx.h + ctx.l) / 2, ctx.adv(40), 9), 10))
    p2 = _rank_cs(_decay_linear(_corr(_ts_rank(ctx.vwap, 4), _ts_rank(ctx.v, 19), 7), 3))
    return p1 / p2


def _a073(ctx: _Ctx) -> pd.DataFrame:
    # (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    #      Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608)
    #                             / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    p1 = _rank_cs(_decay_linear(_delta(ctx.vwap, 5), 3))
    blend = ctx.o * 0.147155 + ctx.l * (1 - 0.147155)
    p2 = _ts_rank(_decay_linear(-_delta(blend, 2) / blend, 3), 17)
    return _max2(p1, p2) * -1


def _a074(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
    #   rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    x = _rank_cs(_corr(ctx.c, _ts_sum(ctx.adv(30), 37), 15))
    y = _rank_cs(_corr(_rank_cs(ctx.h * 0.0261661 + ctx.vwap * (1 - 0.0261661)),
                       ctx.rank("volume"), 11))
    return _lt(x, y) * -1


def _a075(ctx: _Ctx) -> pd.DataFrame:
    # (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
    return _lt(_rank_cs(_corr(ctx.vwap, ctx.v, 4)),
               _rank_cs(_corr(ctx.rank("low"), ctx.rank_of(ctx.adv(50)), 12)))


def _a076(ctx: _Ctx) -> pd.DataFrame:
    # (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
    #      Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha076", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _rank_cs(_decay_linear(_delta(ctx.vwap, 1), 12))
    p2 = _ts_rank(_decay_linear(
        _ts_rank(_corr(_indneutralize(ctx.l, ctx.industry), ctx.adv(81), 8), 20), 17), 19)
    return _max2(p1, p2) * -1


def _a077(ctx: _Ctx) -> pd.DataFrame:
    # (min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    #      rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125))))
    # 注：(((high+low)/2 + high) - (vwap+high)) = (high+low)/2 - vwap
    p1 = _rank_cs(_decay_linear((ctx.h + ctx.l) / 2 - ctx.vwap, 20))
    p2 = _rank_cs(_decay_linear(_corr((ctx.h + ctx.l) / 2, ctx.adv(40), 3), 6))
    return _min2(p1, p2)


def _a078(ctx: _Ctx) -> pd.DataFrame:
    # (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    #                    sum(adv40, 19.7428), 6.83313))^
    #  rank(correlation(rank(vwap), rank(volume), 5.77492)))
    x = _ts_sum(ctx.l * 0.352233 + ctx.vwap * (1 - 0.352233), 20)
    y = _ts_sum(ctx.adv(40), 20)
    base = _rank_cs(_corr(x, y, 7))
    exp = _rank_cs(_corr(ctx.rank("vwap"), ctx.rank("volume"), 6))
    return _signedpow(base, exp)


def _a079(ctx: _Ctx) -> pd.DataFrame:
    # (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) <
    #  rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
    if ctx.industry is None:
        ctx.warn_missing("alpha079", "行业分类（industry）")
        return _nan_like(ctx.c)
    x = _rank_cs(_delta(_indneutralize(ctx.c * 0.60733 + ctx.o * (1 - 0.60733),
                                       ctx.industry), 1))
    y = _rank_cs(_corr(_ts_rank(ctx.vwap, 4), _ts_rank(ctx.adv(150), 9), 15))
    return _lt(x, y)


def _a080(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^
    #   Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha080", "行业分类（industry）")
        return _nan_like(ctx.c)
    base = _rank_cs(_sign(_delta(_indneutralize(ctx.o * 0.868128 + ctx.h * (1 - 0.868128),
                                                ctx.industry), 4)))
    exp = _ts_rank(_corr(ctx.h, ctx.adv(10), 5), 6)
    return _signedpow(base, exp) * -1


def _a081(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) <
    #   rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    inner = _signedpow(_rank_cs(_corr(ctx.vwap, _ts_sum(ctx.adv(10), 50), 8)), 4.0)
    x = _rank_cs(_log(_product(inner, 15)))
    y = _rank_cs(_corr(ctx.rank("vwap"), ctx.rank("volume"), 5))
    return _lt(x, y) * -1


def _a082(ctx: _Ctx) -> pd.DataFrame:
    # (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
    #      Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector),
    #         ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    # 注：open*0.634196 + open*(1-0.634196) = open（论文原式的自加权恒等式）
    if ctx.industry is None:
        ctx.warn_missing("alpha082", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _rank_cs(_decay_linear(_delta(ctx.o, 1), 15))
    p2 = _ts_rank(_decay_linear(_corr(_indneutralize(ctx.v, ctx.industry),
                                      ctx.o, 17), 7), 13)
    return _min2(p1, p2) * -1


def _a083(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) /
    #  (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    sma5 = _ts_sum(ctx.c, 5) / 5
    ratio = (ctx.h - ctx.l) / sma5
    num = _rank_cs(_delay(ratio, 2)) * _rank_cs(_rank_cs(ctx.v))
    return num / (ratio / (ctx.vwap - ctx.c))


def _a084(ctx: _Ctx) -> pd.DataFrame:
    # SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    base = _ts_rank(ctx.vwap - _ts_max(ctx.vwap, 15), 21)
    return _signedpow(base, _delta(ctx.c, 5))


def _a085(ctx: _Ctx) -> pd.DataFrame:
    # (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^
    #  rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
    base = _rank_cs(_corr(ctx.h * 0.876703 + ctx.c * (1 - 0.876703), ctx.adv(30), 10))
    exp = _rank_cs(_corr(_ts_rank((ctx.h + ctx.l) / 2, 4), _ts_rank(ctx.v, 10), 7))
    return _signedpow(base, exp)


def _a086(ctx: _Ctx) -> pd.DataFrame:
    # ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
    #   rank(((open + close) - (vwap + open)))) * -1)
    # 注：((open + close) - (vwap + open)) = close - vwap
    x = _ts_rank(_corr(ctx.c, _ts_sum(ctx.adv(20), 15), 6), 20)
    y = _rank_cs(ctx.c - ctx.vwap)
    return _lt(x, y) * -1


def _a087(ctx: _Ctx) -> pd.DataFrame:
    # (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)),
    #      Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha087", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _rank_cs(_decay_linear(_delta(ctx.c * 0.369701 + ctx.vwap * (1 - 0.369701), 2), 3))
    p2 = _ts_rank(_decay_linear(_abs(_corr(_indneutralize(ctx.adv(81), ctx.industry),
                                           ctx.c, 13)), 5), 14)
    return _max2(p1, p2) * -1


def _a088(ctx: _Ctx) -> pd.DataFrame:
    # (min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)),
    #      Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957)))
    p1 = _rank_cs(_decay_linear(
        (ctx.rank("open") + ctx.rank("low")) - (ctx.rank("high") + ctx.rank("close")), 8))
    p2 = _ts_rank(_decay_linear(_corr(_ts_rank(ctx.c, 8), _ts_rank(ctx.adv(60), 21), 8), 7), 3)
    return _min2(p1, p2)


def _a089(ctx: _Ctx) -> pd.DataFrame:
    # (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744)
    #  - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
    # 注：low*0.967285 + low*(1-0.967285) = low
    if ctx.industry is None:
        ctx.warn_missing("alpha089", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _ts_rank(_decay_linear(_corr(ctx.l, ctx.adv(10), 7), 6), 4)
    p2 = _ts_rank(_decay_linear(_delta(_indneutralize(ctx.vwap, ctx.industry), 3), 10), 15)
    return p1 - p2


def _a090(ctx: _Ctx) -> pd.DataFrame:
    # ((rank((close - ts_max(close, 4.66719)))^
    #   Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha090", "行业分类（industry）")
        return _nan_like(ctx.c)
    base = _rank_cs(ctx.c - _ts_max(ctx.c, 5))
    exp = _ts_rank(_corr(_indneutralize(ctx.adv(40), ctx.industry), ctx.l, 5), 3)
    return _signedpow(base, exp) * -1


def _a091(ctx: _Ctx) -> pd.DataFrame:
    # ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667)
    #   - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha091", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _ts_rank(_decay_linear(
        _decay_linear(_corr(_indneutralize(ctx.c, ctx.industry), ctx.v, 10), 16), 4), 5)
    p2 = _rank_cs(_decay_linear(_corr(ctx.vwap, ctx.adv(30), 4), 3))
    return (p1 - p2) * -1


def _a092(ctx: _Ctx) -> pd.DataFrame:
    # (min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683),
    #      Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584)))
    p1 = _ts_rank(_decay_linear(
        (((ctx.h + ctx.l) / 2 + ctx.c) < (ctx.l + ctx.o)).astype(float), 15), 19)
    p2 = _ts_rank(_decay_linear(
        _corr(ctx.rank("low"), ctx.rank_of(ctx.adv(30)), 8), 7), 7)
    return _min2(p1, p2)


def _a093(ctx: _Ctx) -> pd.DataFrame:
    # (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455)
    #  / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    if ctx.industry is None:
        ctx.warn_missing("alpha093", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _ts_rank(_decay_linear(_corr(_indneutralize(ctx.vwap, ctx.industry),
                                      ctx.adv(81), 17), 20), 8)
    p2 = _rank_cs(_decay_linear(
        _delta(ctx.c * 0.524434 + ctx.vwap * (1 - 0.524434), 3), 16))
    return p1 / p2


def _a094(ctx: _Ctx) -> pd.DataFrame:
    # ((rank((vwap - ts_min(vwap, 11.5783)))^
    #   Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    base = _rank_cs(ctx.vwap - _ts_min(ctx.vwap, 12))
    exp = _ts_rank(_corr(_ts_rank(ctx.vwap, 20), _ts_rank(ctx.adv(60), 4), 18), 3)
    return _signedpow(base, exp) * -1


def _a095(ctx: _Ctx) -> pd.DataFrame:
    # (rank((open - ts_min(open, 12.4105))) <
    #  Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    x = _rank_cs(ctx.o - _ts_min(ctx.o, 12))
    corr = _corr(_ts_sum((ctx.h + ctx.l) / 2, 19), _ts_sum(ctx.adv(40), 19), 13)
    y = _ts_rank(_rank_cs(corr) ** 5, 12)
    return _lt(x, y)


def _a096(ctx: _Ctx) -> pd.DataFrame:
    # (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
    #      Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    p1 = _ts_rank(_decay_linear(_corr(ctx.rank("vwap"), ctx.rank("volume"), 4), 4), 8)
    p2 = _ts_rank(_decay_linear(
        _ts_argmax(_corr(_ts_rank(ctx.c, 7), _ts_rank(ctx.adv(60), 4), 4), 13), 14), 13)
    return _max2(p1, p2) * -1


def _a097(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523))
    #   - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    if ctx.industry is None:
        ctx.warn_missing("alpha097", "行业分类（industry）")
        return _nan_like(ctx.c)
    p1 = _rank_cs(_decay_linear(
        _delta(_indneutralize(ctx.l * 0.721001 + ctx.vwap * (1 - 0.721001),
                              ctx.industry), 3), 20))
    p2 = _ts_rank(_decay_linear(
        _ts_rank(_corr(_ts_rank(ctx.l, 8), _ts_rank(ctx.adv(60), 17), 5), 19), 16), 7)
    return (p1 - p2) * -1


def _a098(ctx: _Ctx) -> pd.DataFrame:
    # (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
    #  - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    p1 = _rank_cs(_decay_linear(_corr(ctx.vwap, _ts_sum(ctx.adv(5), 26), 5), 7))
    p2 = _rank_cs(_decay_linear(
        _ts_rank(_ts_argmin(_corr(ctx.rank("open"), ctx.rank_of(ctx.adv(15)), 21), 9), 7), 8))
    return p1 - p2


def _a099(ctx: _Ctx) -> pd.DataFrame:
    # ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <
    #   rank(correlation(low, volume, 6.28259))) * -1)
    x = _rank_cs(_corr(_ts_sum((ctx.h + ctx.l) / 2, 20), _ts_sum(ctx.adv(60), 20), 9))
    y = _rank_cs(_corr(ctx.l, ctx.v, 6))
    return _lt(x, y) * -1


def _a100(ctx: _Ctx) -> pd.DataFrame:
    # (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)),
    #     IndClass.subindustry), IndClass.subindustry)))
    #   - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry)))
    #   * (volume / adv20))))
    if ctx.industry is None:
        ctx.warn_missing("alpha100", "行业分类（industry）")
        return _nan_like(ctx.c)
    x = ((ctx.c - ctx.l) - (ctx.h - ctx.c)) / (ctx.h - ctx.l) * ctx.v
    p1 = 1.5 * _scale(_indneutralize(_indneutralize(_rank_cs(x), ctx.industry),
                                     ctx.industry))
    p2 = _scale(_indneutralize(
        _corr(ctx.c, _rank_cs(ctx.adv(20)), 5) - _rank_cs(_ts_argmin(ctx.c, 30)),
        ctx.industry))
    return -(p1 - p2) * (ctx.v / ctx.adv(20))


def _a101(ctx: _Ctx) -> pd.DataFrame:
    # ((close - open) / ((high - low) + .001))
    return (ctx.c - ctx.o) / ((ctx.h - ctx.l) + 0.001)


# 注册表（顺序与论文编号一致）
_ALPHA_FUNCS = {name: globals()[f"_a{name[-3:]}"] for name in ALPHA101_NAMES}
assert len(_ALPHA_FUNCS) == 101


# ---------------------------------------------------------------------------
# Alpha158：Qlib 语义（滚动算子 min_periods=1，逐条对齐 qlib）
# ---------------------------------------------------------------------------

def _qlib_rolling(a: np.ndarray, w: int, func: str) -> np.ndarray:
    """qlib Rolling 算子：pandas rolling(w, min_periods=1)，部分窗口从第
    一天起就有值（qlib 语义，见模块 docstring）。"""
    return getattr(pd.Series(a).rolling(w, min_periods=1), func)().to_numpy()


def _qlib_idxmax(a: np.ndarray, w: int) -> np.ndarray:
    """qlib IdxMax：窗口内最大值位置（1 起；部分窗口按已有元素）。"""
    n = len(a)
    out = np.full(n, np.nan)
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        out[w - 1:] = np.argmax(win, axis=1) + 1
    for t in range(min(w - 1, n)):
        out[t] = np.argmax(a[:t + 1]) + 1
    return out


def _qlib_idxmin(a: np.ndarray, w: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan)
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        out[w - 1:] = np.argmin(win, axis=1) + 1
    for t in range(min(w - 1, n)):
        out[t] = np.argmin(a[:t + 1]) + 1
    return out


def _qlib_slope_rsqr_resi(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """qlib 的 Slope/Rsquare/Resi（data/_libs/rolling.pyx 的闭式等价形式）。

    qlib 对窗口内 N 个有效值做 y 对位置 x 的 OLS，最新值的 x=w，其余依次
    递减——即 x 是"窗口内 0..N-1 序号"的仿射变换（斜率 +1），故回归斜率、
    R² 与"对序号 i=0..N-1 回归"完全一致：
      slope   = (N·Σ(i·y) − Σi·Σy) / (N·Σi² − (Σi)²)
      rsquare = corr(i, y)²
      resi    = (y_last − ȳ) − slope·(N−1)/2
    Σ(i·y) 用卷积求（权重 = 距今的天数），滚动和用 cumsum 差分。
    """
    n = len(a)
    idx = np.arange(n)
    nwin = np.minimum(idx + 1, w)
    cs_y = np.cumsum(a)
    cs_y2 = np.cumsum(a * a)
    prev_y = np.concatenate([np.zeros(w), cs_y])[:n]
    prev_y2 = np.concatenate([np.zeros(w), cs_y2])[:n]
    wy = cs_y - prev_y          # 窗口内 Σy
    wy2 = cs_y2 - prev_y2       # 窗口内 Σy²
    wi = np.convolve(a, np.arange(w, dtype=float), mode="full")[:n]  # Σ(年龄·y)
    s1 = nwin * (nwin - 1) / 2.0
    s2 = nwin * (nwin - 1) * (2 * nwin - 1) / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        den = nwin * s2 - s1 * s1
        # wi 按"距今年龄"加权；qlib 的 x 与年龄反向（最新值 x=w），
        # 故斜率取负（与 np.polyfit 对窗口内序号 0..N-1 回归一致）
        slope = -(nwin * wi - s1 * wy) / den
        num = (nwin * wi - s1 * wy) ** 2
        rsqr = num / (den * np.maximum(nwin * wy2 - wy * wy, 0.0))
        resi = (a - wy / nwin) - slope * (nwin - 1) / 2.0
    return slope, rsqr, resi


def _qlib_corr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    """qlib Corr：rolling corr（min_periods=1），且任一侧滚动 std(ddof=1)
    与 0 的距离 < 2e-5 时置 NaN（qlib ops.py 的 Corr 后处理）。"""
    n = len(a)
    nwin = np.minimum(np.arange(n) + 1, w)
    def wsum(z: np.ndarray) -> np.ndarray:
        cs = np.cumsum(z)
        return cs - np.concatenate([np.zeros(w), cs])[:n]
    wa, wb = wsum(a), wsum(b)
    wab, wa2, wb2 = wsum(a * b), wsum(a * a), wsum(b * b)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = ((nwin * wab - wa * wb)
                / np.sqrt((nwin * wa2 - wa ** 2) * (nwin * wb2 - wb ** 2)))
    std_a = pd.Series(a).rolling(w, min_periods=1).std().to_numpy()
    std_b = pd.Series(b).rolling(w, min_periods=1).std().to_numpy()
    corr[np.isclose(std_a, 0, atol=2e-05) | np.isclose(std_b, 0, atol=2e-05)] = np.nan
    return corr


def compute_alpha158(frames: dict[str, pd.DataFrame],
                     include: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """按 qlib Alpha158 默认配置计算 158 个量价特征（宽表 × 宽表）。

    参数同 compute_alpha101。返回 {特征名: 日期×股票 DataFrame}。
    include 指定时**只计算**被请求的特征（跳过其余算子的构造，
    大股票池下可显著降低内存与耗时）。
    """
    o, h, l, c, v = (frames["open"], frames["high"], frames["low"],
                     frames["close"], frames["volume"])
    vwap = frames["vwap"]
    out: dict[str, pd.DataFrame] = {}
    if include is not None:
        unknown = set(include) - set(ALPHA158_NAMES)
        if unknown:
            raise ValueError(f"未知 Alpha158 特征: {sorted(unknown)}")
        want = set(include)
    else:
        want = set(ALPHA158_NAMES)

    def need(key: str) -> bool:
        return key in want

    # (a) kbar 特征（Qlib "kbar" 配置；Greater/Less 即 max/min）
    if need("KMID"):
        out["KMID"] = (c - o) / o
    if need("KLEN"):
        out["KLEN"] = (h - l) / o
    if need("KMID2"):
        out["KMID2"] = (c - o) / (h - l + 1e-12)
    if need("KUP"):
        out["KUP"] = (h - np.maximum(o, c)) / o
    if need("KUP2"):
        out["KUP2"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
    if need("KLOW"):
        out["KLOW"] = (np.minimum(o, c) - l) / o
    if need("KLOW2"):
        out["KLOW2"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
    if need("KSFT"):
        out["KSFT"] = (2 * c - h - l) / o
    if need("KSFT2"):
        out["KSFT2"] = (2 * c - h - l) / (h - l + 1e-12)

    # (b) 原始价格特征（Qlib handler.py 默认：window=0 → 当日值 / $close）
    if need("OPEN0"):
        out["OPEN0"] = o / c
    if need("HIGH0"):
        out["HIGH0"] = h / c
    if need("LOW0"):
        out["LOW0"] = l / c
    if need("VWAP0"):
        out["VWAP0"] = vwap / c

    # (c) 滚动算子 × 5 窗口（Qlib "rolling" 默认配置，29 种）
    # 滞后/差分均按个股自身交易日（_shift_np），跳过停牌日
    for w in ALPHA158_WINDOWS:
        if need(f"ROC{w}"):
            out[f"ROC{w}"] = _map_cols(c, lambda a: _shift_np(a, w)) / c
        if need(f"MA{w}"):
            out[f"MA{w}"] = _map_cols(c, lambda a: _qlib_rolling(a, w, "mean")) / c
        if need(f"STD{w}"):
            out[f"STD{w}"] = _map_cols(c, lambda a: _qlib_rolling(a, w, "std")) / c
        if need(f"BETA{w}"):
            out[f"BETA{w}"] = _map_cols(
                c, lambda a: _qlib_slope_rsqr_resi(a, w)[0]) / c
        if need(f"RSQR{w}"):
            rsqr = _map_cols(c, lambda a: _qlib_slope_rsqr_resi(a, w)[1])
            # qlib：滚动 std(ddof=1)≈0 的窗口 R² 置 NaN
            std_zero = _map_cols(c, lambda a: np.isclose(
                pd.Series(a).rolling(w, min_periods=1).std().to_numpy(),
                0, atol=2e-05))
            out[f"RSQR{w}"] = rsqr.mask(std_zero.astype(bool))
        if need(f"RESI{w}"):
            out[f"RESI{w}"] = _map_cols(
                c, lambda a: _qlib_slope_rsqr_resi(a, w)[2]) / c
        if need(f"MAX{w}"):
            out[f"MAX{w}"] = _map_cols(h, lambda a: _qlib_rolling(a, w, "max")) / c
        if need(f"MIN{w}"):
            out[f"MIN{w}"] = _map_cols(l, lambda a: _qlib_rolling(a, w, "min")) / c
        if need(f"QTLU{w}"):
            out[f"QTLU{w}"] = _map_cols(
                c, lambda a: pd.Series(a).rolling(w, min_periods=1)
                .quantile(0.8).to_numpy()) / c
        if need(f"QTLD{w}"):
            out[f"QTLD{w}"] = _map_cols(
                c, lambda a: pd.Series(a).rolling(w, min_periods=1)
                .quantile(0.2).to_numpy()) / c
        if need(f"RANK{w}"):
            out[f"RANK{w}"] = _map_cols(
                c, lambda a: pd.Series(a).rolling(w, min_periods=1)
                .rank(pct=True).to_numpy())
        if need(f"RSV{w}"):
            min_low = _map_cols(l, lambda a: _qlib_rolling(a, w, "min"))
            max_high = _map_cols(h, lambda a: _qlib_rolling(a, w, "max"))
            out[f"RSV{w}"] = (c - min_low) / (max_high - min_low + 1e-12)
        if need(f"IMAX{w}"):
            out[f"IMAX{w}"] = _map_cols(h, lambda a: _qlib_idxmax(a, w)) / w
        if need(f"IMIN{w}"):
            out[f"IMIN{w}"] = _map_cols(l, lambda a: _qlib_idxmin(a, w)) / w
        if need(f"IMXD{w}"):
            out[f"IMXD{w}"] = (_map_cols(h, lambda a: _qlib_idxmax(a, w))
                               - _map_cols(l, lambda a: _qlib_idxmin(a, w))) / w

        if need(f"CORR{w}"):
            log_v1 = _log(v + 1)
            out[f"CORR{w}"] = _map_pair(c, log_v1,
                                        lambda a, b: _qlib_corr(a, b, w))
        if need(f"CORD{w}"):
            close_ret = _map_cols(c, _pct_change_np)
            vol_ret = _map_cols(v, _pct_change_np)
            out[f"CORD{w}"] = _map_pair(close_ret, _log(vol_ret + 1),
                                        lambda a, b: _qlib_corr(a, b, w))
        if any(need(k) for k in (f"CNTP{w}", f"CNTN{w}", f"CNTD{w}")):
            up = _map_cols(c, lambda a: (a > _shift_np(a, 1)).astype(float))
            down = _map_cols(c, lambda a: (a < _shift_np(a, 1)).astype(float))
            if need(f"CNTP{w}"):
                out[f"CNTP{w}"] = _map_cols(up, lambda a: _qlib_rolling(a, w, "mean"))
            if need(f"CNTN{w}"):
                out[f"CNTN{w}"] = _map_cols(down, lambda a: _qlib_rolling(a, w, "mean"))
            if need(f"CNTD{w}"):
                out[f"CNTD{w}"] = (out[f"CNTP{w}"] if f"CNTP{w}" in out
                                   else _map_cols(up, lambda a: _qlib_rolling(a, w, "mean"))) \
                    - (out[f"CNTN{w}"] if f"CNTN{w}" in out
                       else _map_cols(down, lambda a: _qlib_rolling(a, w, "mean")))

        if any(need(k) for k in (f"SUMP{w}", f"SUMN{w}", f"SUMD{w}")):
            diff = _map_cols(c, lambda a: a - _shift_np(a, 1))
            abs_diff = _abs(diff)
            denom = _map_cols(abs_diff, lambda a: _qlib_rolling(a, w, "sum")) + 1e-12
            if need(f"SUMP{w}"):
                out[f"SUMP{w}"] = _map_cols(
                    (diff > 0).astype(float),
                    lambda a: _qlib_rolling(a, w, "sum")) / denom
            if need(f"SUMN{w}"):
                out[f"SUMN{w}"] = _map_cols(
                    (diff < 0).astype(float),
                    lambda a: _qlib_rolling(a, w, "sum")) / denom
            if need(f"SUMD{w}"):
                out[f"SUMD{w}"] = (out[f"SUMP{w}"] if f"SUMP{w}" in out
                                   else _map_cols((diff > 0).astype(float),
                                                  lambda a: _qlib_rolling(a, w, "sum")) / denom) \
                    - (out[f"SUMN{w}"] if f"SUMN{w}" in out
                       else _map_cols((diff < 0).astype(float),
                                      lambda a: _qlib_rolling(a, w, "sum")) / denom)

        if need(f"VMA{w}"):
            out[f"VMA{w}"] = (_map_cols(v, lambda a: _qlib_rolling(a, w, "mean"))
                              / (v + 1e-12))
        if need(f"VSTD{w}"):
            out[f"VSTD{w}"] = (_map_cols(v, lambda a: _qlib_rolling(a, w, "std"))
                               / (v + 1e-12))
        if need(f"WVMA{w}"):
            close_ret = _map_cols(c, _pct_change_np)
            wv = _abs(close_ret - 1) * v
            wv_mean = _map_cols(wv, lambda a: _qlib_rolling(a, w, "mean"))
            wv_std = _map_cols(wv, lambda a: _qlib_rolling(a, w, "std"))
            out[f"WVMA{w}"] = wv_std / (wv_mean + 1e-12)

        if any(need(k) for k in (f"VSUMP{w}", f"VSUMN{w}", f"VSUMD{w}")):
            vdiff = _map_cols(v, lambda a: a - _shift_np(a, 1))
            vabs = _abs(vdiff)
            vden = _map_cols(vabs, lambda a: _qlib_rolling(a, w, "sum")) + 1e-12
            if need(f"VSUMP{w}"):
                out[f"VSUMP{w}"] = _map_cols(
                    (vdiff > 0).astype(float),
                    lambda a: _qlib_rolling(a, w, "sum")) / vden
            if need(f"VSUMN{w}"):
                out[f"VSUMN{w}"] = _map_cols(
                    (vdiff < 0).astype(float),
                    lambda a: _qlib_rolling(a, w, "sum")) / vden
            if need(f"VSUMD{w}"):
                out[f"VSUMD{w}"] = (out[f"VSUMP{w}"] if f"VSUMP{w}" in out
                                    else _map_cols((vdiff > 0).astype(float),
                                                   lambda a: _qlib_rolling(a, w, "sum")) / vden) \
                    - (out[f"VSUMN{w}"] if f"VSUMN{w}" in out
                       else _map_cols((vdiff < 0).astype(float),
                                      lambda a: _qlib_rolling(a, w, "sum")) / vden)

    if include is not None:
        out = {k: out[k] for k in include}
    return {k: _clean_inf(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def prepare_frames(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """把长表（date, symbol, open, high, low, close, volume, amount）
    转成宽表字典 {open/high/low/close/volume/vwap/returns: 日期 × 股票}。

    vwap = amount / volume；returns = close.pct_change()。amount 缺失时
    vwap 全 NaN（依赖 vwap 的因子相应为 NaN，如实声明）。
    """
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}")
    df = long_df.sort_values(["symbol", "date"])
    frames = {}
    for field in ("open", "high", "low", "close", "volume"):
        frames[field] = df.pivot_table(
            index="date", columns="symbol", values=field, aggfunc="last")
    frames["vwap"] = (df.pivot_table(index="date", columns="symbol",
                                     values="amount", aggfunc="last")
                      / frames["volume"])
    # returns 按个股自身交易日计算（跳过停牌日，与 _delay/_delta 同口径）
    frames["returns"] = _map_cols(frames["close"], _pct_change_np)
    return frames


def compute_alpha101(frames: dict[str, pd.DataFrame],
                     cap: pd.Series | None = None,
                     industry: pd.Series | None = None,
                     include: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """计算 101 个 WorldQuant alpha（宽表 × 宽表）。

    参数
    ----
    frames : prepare_frames 输出的宽表字典（open/high/low/close/volume/
             vwap/returns，均以日期为行、股票代码为列）。
    cap : 可选，symbol → 市值 Series（alpha056 需要）。
    industry : 可选，symbol → 行业标签 Series（18 个行业 alpha 需要；
               论文的 sector/industry/subindustry 三级分类统一用此单一
               分类近似）。
    include : 可选，只计算指定 alpha（名称如 "alpha001"）。

    返回 {alpha 名: 日期 × 股票 DataFrame}；±inf 统一替换为 NaN。
    """
    names = include if include is not None else ALPHA101_NAMES
    unknown = set(names) - set(_ALPHA_FUNCS)
    if unknown:
        raise ValueError(f"未知 alpha: {sorted(unknown)}")
    ctx = _Ctx(frames, cap=cap, industry=industry)
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        out[name] = _clean_inf(_ALPHA_FUNCS[name](ctx))
    return out


# ---------------------------------------------------------------------------
# 数据加载（复用第 1/6 步的 CSV 口径）
# ---------------------------------------------------------------------------

def load_frames(symbols: list[str], datadir: str = "data/daily",
                adjust: str = "qfq", start: str | None = None,
                end: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """从 data/daily/<代码>_<adjust>.csv 逐票读取并直接构建宽表字典。

    逐票读入为 Series、最后用 dict 一次性对齐成宽表（并集日期索引），
    **不做全量长表 concat + pivot_table**：内存占用与股票数成正比，
    与总行数无关（400 只 × 约 1700 日峰值仅数十 MB）。

    返回 (frames, names)：names 为 symbol → 股票名称（CSV 无名称时为空）。
    列口径经 backtest.standardize_stock 归一（date/open/high/low/close/
    volume/amount）。
    """
    fields: dict[str, dict[str, pd.Series]] = {
        f: {} for f in ("open", "high", "low", "close", "volume", "amount")}
    names: dict[str, str] = {}
    missing = 0
    for symbol in symbols:
        path = Path(datadir) / f"{symbol}_{adjust}.csv"
        if not path.exists():
            missing += 1
            continue
        df = standardize_stock(pd.read_csv(path))
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        if df.empty:
            missing += 1
            continue
        df = df.set_index("date")
        for f in fields:
            if f in df.columns:
                fields[f][symbol] = df[f]
    if not fields["close"]:
        raise RuntimeError(
            f"{datadir} 下没有任何可用日线（--symbols/--pool 参数或先执行 "
            f"fetch_daily.py / screener.py fetch）；缺失 {missing} 只")
    if missing:
        print(f"[warn] {missing} 只股票无可用日线，已跳过", file=sys.stderr)
    if not fields["amount"]:
        print("[warn] 数据缺少成交额列，vwap 全为 NaN（依赖 vwap 的因子"
              "相应为 NaN）", file=sys.stderr)
        fields["amount"] = {s: pd.Series(dtype=float) for s in fields["close"]}
    frames = {f: pd.DataFrame(fields[f]).sort_index()
              for f in ("open", "high", "low", "close", "volume", "amount")}
    frames["vwap"] = frames["amount"] / frames["volume"]
    frames["returns"] = _map_cols(frames["close"], _pct_change_np)
    del frames["amount"]
    return frames, names


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def latest_cross_section(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每只股票取其自身最后一行的因子值（股票 × 因子）。"""
    rows = []
    for name, frame in result.items():
        s = frame.apply(lambda col: col.dropna().iloc[-1] if col.notna().any()
                        else np.nan)
        rows.append(pd.Series(s, name=name))
    return pd.concat(rows, axis=1)


def coverage_summary(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每个因子的有效值占比与描述统计（因子 × 指标）。"""
    rows = []
    for name, frame in result.items():
        vals = frame.to_numpy(dtype=float)
        finite = np.isfinite(vals)
        rows.append({
            "factor": name,
            "n_finite": int(finite.sum()),
            "coverage": float(finite.mean()),
            "mean": float(np.nanmean(vals[finite])) if finite.any() else np.nan,
            "std": float(np.nanstd(vals[finite])) if finite.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("coverage", ascending=False)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def _synthetic_frames(n_stocks: int, n_days: int, seed: int) -> dict[str, pd.DataFrame]:
    """合成日线宽表（自检/演示用）：几何随机游走 + 自洽 OHLC。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    symbols = [f"s{i:03d}" for i in range(n_stocks)]
    rets = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
    close = 10.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * (1 + rng.normal(0, 0.005, size=(n_days, n_stocks)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=(n_days, n_stocks))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=(n_days, n_stocks))))
    volume = np.exp(rng.normal(15.0, 0.5, size=(n_days, n_stocks)))
    vwap = close * (1 + rng.normal(0, 0.002, size=(n_days, n_stocks)))
    amount = vwap * volume
    def frame(arr):
        return pd.DataFrame(arr, index=dates, columns=symbols)
    return {
        "open": frame(open_), "high": frame(high), "low": frame(low),
        "close": frame(close), "volume": frame(volume), "vwap": frame(vwap),
        "returns": frame(close).pct_change(), "amount": frame(amount),
    }


def _selftest() -> int:
    """合成数据自检：形状/取值断言 + 无未来函数两条性质检验。"""
    print("== alpha_factors selftest ==")
    frames = _synthetic_frames(5, 260, seed=42)
    like = frames["close"]

    # --- 1. 形状与名单 ---
    a101 = compute_alpha101(frames)
    assert set(a101) == set(ALPHA101_NAMES), "alpha101 名单不完整"
    for name, f in a101.items():
        assert f.shape == like.shape, f"{name} 形状错误"
        assert not np.isinf(f.to_numpy()).any(), f"{name} 含 ±inf"
    print(f"[ok] Alpha101: {len(a101)} 个，形状 {like.shape}，无 ±inf")

    a158 = compute_alpha158(frames)
    assert list(a158) == ALPHA158_NAMES, "alpha158 名单/顺序不符"
    for name, f in a158.items():
        assert f.shape == like.shape, f"{name} 形状错误"
        assert not np.isinf(f.to_numpy()).any(), f"{name} 含 ±inf"
    print(f"[ok] Alpha158: {len(a158)} 个（9 K线 + 4 原始价 + 29×5 滚动），无 ±inf")

    # --- 2. 无未来函数（截断一致性）---
    cut = {k: v.iloc[:200] for k, v in frames.items()}
    short101 = compute_alpha101(cut)
    short158 = compute_alpha158(cut)
    t0 = 199
    for name in ALPHA101_NAMES:
        assert np.allclose(a101[name].iloc[t0], short101[name].iloc[t0],
                           equal_nan=True), f"{name} 截断不一致"
    for name in ALPHA158_NAMES:
        assert np.allclose(a158[name].iloc[t0], short158[name].iloc[t0],
                           equal_nan=True), f"{name} 截断不一致"
    print("[ok] 截断一致性（t=199 处与完整计算逐值相等）：101 + 158 全部通过")

    # --- 3. 无未来函数（未来扰动不变性）---
    perturbed = {k: v.copy() for k, v in frames.items()}
    for k in ("open", "high", "low", "close", "vwap"):
        perturbed[k].iloc[220:] *= 1.5
    perturbed["volume"].iloc[220:] *= 10.0
    perturbed["returns"] = perturbed["close"].pct_change()
    b101 = compute_alpha101(perturbed)
    b158 = compute_alpha158(perturbed)
    for name in ("alpha001", "alpha004", "alpha012", "alpha026", "alpha032",
                 "alpha060", "alpha101"):
        assert np.allclose(a101[name].iloc[:219], b101[name].iloc[:219],
                           equal_nan=True), f"{name} 受未来数据影响"
    for name in ("ROC5", "MA20", "BETA60", "RSQR10", "RESI30", "CORR20",
                 "CORD60", "CNTP5", "SUMP20", "VMA5", "WVMA60", "VSUMD30",
                 "RSV10", "RANK20", "IMXD10", "KMID", "VWAP0"):
        assert np.allclose(a158[name].iloc[:219], b158[name].iloc[:219],
                           equal_nan=True), f"{name} 受未来数据影响"
    print("[ok] 未来扰动不变性（前 219 行逐值相等）")

    # --- 3b. 停牌缺口（并集日历 NaN）→ 按个股自身交易日计算 ---
    gappy = {k: v.copy() for k, v in frames.items()}
    for k in ("open", "high", "low", "close", "vwap", "volume"):
        gappy[k].iloc[50:54, 1] = np.nan   # s001 停牌 4 个交易日
        gappy[k].iloc[80:82, 0] = np.nan   # s000 停牌 2 个交易日
    gappy["returns"] = _map_cols(gappy["close"], _pct_change_np)
    g101 = compute_alpha101(gappy)
    g158 = compute_alpha158(gappy)
    for name in ("alpha012", "alpha026", "alpha032", "alpha101"):
        assert np.isnan(g101[name].iloc[50:54, 1]).all(), f"{name} 停牌行应为 NaN"
        assert np.isnan(g101[name].iloc[80:82, 0]).all(), f"{name} 停牌行应为 NaN"
    for name in ("KMID", "ROC5", "CNTP5", "VMA5"):
        assert np.isnan(g158[name].iloc[50:54, 1]).all(), f"{name} 停牌行应为 NaN"
        assert np.isnan(g158[name].iloc[80:82, 0]).all(), f"{name} 停牌行应为 NaN"
    # 复牌后的值与"去掉停牌行后的连续日历"上计算的逐值一致
    drop_idx = frames["close"].index[50:54].union(frames["close"].index[80:82])
    comp = {k: v.drop(index=drop_idx) for k, v in frames.items()}
    comp["returns"] = _map_cols(comp["close"], _pct_change_np)
    c101 = compute_alpha101(comp)
    c158 = compute_alpha158(comp)
    for name in ("alpha012", "alpha026", "alpha032", "alpha101"):
        got = g101[name]["s001"].loc[frames["close"].index[54]]
        want = c101[name]["s001"].loc[frames["close"].index[54]]
        assert np.allclose(got, want, equal_nan=True), \
            f"{name} 复牌后首个值与连续日历不一致"
    for name in ("ROC5", "CNTP5", "VMA5", "MA20"):
        got = g158[name]["s000"].loc[frames["close"].index[82]]
        want = c158[name]["s000"].loc[frames["close"].index[82]]
        assert np.allclose(got, want, equal_nan=True), \
            f"{name} 复牌后首个值与连续日历不一致"
    print("[ok] 停牌缺口：停牌行 NaN，复牌后按个股自身交易日逐值一致")

    # --- 4. 手工精确值（2 股票 × 6 日，显式数字）---
    dates = pd.date_range("2020-01-02", periods=6, freq="B")
    syms = ["x", "y"]
    o = pd.DataFrame([[10.0, 20.0], [10.2, 20.4], [10.1, 20.2],
                      [10.5, 20.9], [10.4, 20.8], [10.6, 21.0]],
                     index=dates, columns=syms)
    c = pd.DataFrame([[10.5, 21.0], [10.3, 20.6], [10.4, 20.8],
                      [10.9, 21.4], [10.6, 21.2], [11.0, 21.6]],
                     index=dates, columns=syms)
    h = pd.DataFrame([[10.8, 21.5], [10.6, 21.0], [10.7, 21.2],
                      [11.2, 22.0], [11.0, 21.8], [11.4, 22.2]],
                     index=dates, columns=syms)
    l = pd.DataFrame([[9.8, 19.8], [10.0, 20.2], [9.9, 20.0],
                      [10.3, 20.6], [10.2, 20.5], [10.5, 20.9]],
                     index=dates, columns=syms)
    v = pd.DataFrame([[100.0, 200.0], [120.0, 240.0], [90.0, 180.0],
                      [150.0, 300.0], [110.0, 220.0], [130.0, 260.0]],
                     index=dates, columns=syms)
    vwap = pd.DataFrame([[10.4, 20.8], [10.25, 20.5], [10.35, 20.7],
                         [10.7, 21.1], [10.5, 21.0], [10.8, 21.3]],
                        index=dates, columns=syms)
    tiny = {"open": o, "high": h, "low": l, "close": c, "volume": v,
            "vwap": vwap, "returns": c.pct_change()}
    t101 = compute_alpha101(tiny)
    t158 = compute_alpha158(tiny)

    # alpha101 = (close - open) / (high - low + 0.001)
    for t, (cc, oo, hh, ll) in enumerate(zip(c["x"], o["x"], h["x"], l["x"])):
        expect = (cc - oo) / (hh - ll + 0.001)
        assert np.isclose(t101["alpha101"]["x"].iloc[t], expect), \
            f"alpha101 第 {t} 行手工值不符"
    # alpha012 = sign(delta(volume,1)) * -delta(close,1)
    for t in range(1, 6):
        expect = np.sign(v["x"].iloc[t] - v["x"].iloc[t - 1]) \
            * -(c["x"].iloc[t] - c["x"].iloc[t - 1])
        assert np.isclose(t101["alpha012"]["x"].iloc[t], expect), \
            f"alpha012 第 {t} 行手工值不符"
    # KMID = (close - open)/open；KUP = (high - max(open,close))/open
    for t in range(6):
        assert np.isclose(t158["KMID"]["x"].iloc[t],
                          (c["x"].iloc[t] - o["x"].iloc[t]) / o["x"].iloc[t])
        assert np.isclose(t158["KUP"]["x"].iloc[t],
                          (h["x"].iloc[t] - max(o["x"].iloc[t], c["x"].iloc[t]))
                          / o["x"].iloc[t])
    # ROC5[t] = close[t-5]/close[t]
    assert np.isclose(t158["ROC5"]["x"].iloc[5], c["x"].iloc[0] / c["x"].iloc[5])
    # VMA5[t] = mean(volume[t-4..t]) / volume[t]
    assert np.isclose(t158["VMA5"]["x"].iloc[4],
                      v["x"].iloc[:5].mean() / v["x"].iloc[4])
    # SUMP5[4] = #(diff>0 的交易日)/Σ|diff|（窗口 0..4；第 0 行 diff 为 NaN，
    # 比较为 False 计入分子 0，分母 rolling sum 跳过 NaN）
    diff = c["x"].diff()
    num = int((diff.iloc[1:5] > 0).sum())
    den = diff.iloc[1:5].abs().sum()
    assert np.isclose(t158["SUMP5"]["x"].iloc[4], num / (den + 1e-12)), \
        f"SUMP5 手工值不符: {t158['SUMP5']['x'].iloc[4]} vs {num/(den+1e-12)}"
    # RANK5[t] = qlib rolling rank(pct=True)（pandas 语义）
    expect_rank = pd.Series(c["x"].iloc[:5]).rank(pct=True).iloc[-1]
    assert np.isclose(t158["RANK5"]["x"].iloc[4], expect_rank), \
        "RANK5 手工值不符"
    # BETA/RSQR/RESI：与 np.polyfit（对窗口内序号 0..N-1 回归）一致
    seg = c["x"].iloc[:5].to_numpy()
    slope_pf = np.polyfit(np.arange(5), seg, 1)[0]
    assert np.isclose(t158["BETA5"]["x"].iloc[4], slope_pf / c["x"].iloc[4]), \
        "BETA5 与 polyfit 不符"
    r_pf = np.corrcoef(np.arange(5), seg)[0, 1] ** 2
    assert np.isclose(t158["RSQR5"]["x"].iloc[4], r_pf), "RSQR5 与 corr² 不符"
    resid_pf = seg[-1] - np.polyval(np.polyfit(np.arange(5), seg, 1), 4)
    assert np.isclose(t158["RESI5"]["x"].iloc[4], resid_pf / c["x"].iloc[4]), \
        "RESI5 与 polyfit 残差不符"
    # IMXD10 的结构断言：IMAX ∈ (0,1]、IMIN ∈ (0,1]
    imax = t158["IMAX5"]["x"].dropna()
    imin = t158["IMIN5"]["x"].dropna()
    assert ((imax > 0) & (imax <= 1)).all() and ((imin > 0) & (imin <= 1)).all()
    print("[ok] 手工精确值（alpha101/012、KMID/KUP/ROC5/VMA5/SUMP5/RANK5/"
          "BETA5/RSQR5/RESI5/IMAX/IMIN）全部通过")

    # --- 5. 横截面 rank 语义 ---
    # 同一日期上，5 只股票的 rank 分位均值 = (m+1)/(2m)
    r = _rank_cs(frames["close"])
    for t in (0, 50, 259):
        vals = r.iloc[t].dropna()
        assert np.isclose(vals.mean(), (len(vals) + 1) / (2 * len(vals))), \
            f"横截面 rank 均值不符 (t={t})"
    print("[ok] 横截面 rank：每行分位秩均值为 (m+1)/(2m)")

    # --- 6. 依赖分类/市值的 alpha ---
    bare = compute_alpha101(frames)
    for name in ALPHA101_NEEDS_INDUSTRY | ALPHA101_NEEDS_CAP:
        assert bare[name].isna().all().all(), f"{name} 未提供输入时应全 NaN"
    ind = pd.Series({s: ("A" if i % 2 == 0 else "B")
                     for i, s in enumerate(like.columns)})
    cap = pd.Series({s: 1e9 + i for i, s in enumerate(like.columns)},
                    dtype=float)
    full = compute_alpha101(frames, cap=cap, industry=ind)
    for name in ALPHA101_NEEDS_INDUSTRY:
        f = full[name]
        assert f.notna().any().any(), f"{name} 提供行业后仍全 NaN"
    f056 = full["alpha056"]
    assert f056.notna().any().any(), "alpha056 提供 cap 后仍全 NaN"
    print("[ok] 行业/市值依赖：19 个 alpha 缺输入全 NaN，提供后有限值")
    print("== selftest 全部通过 ==")
    return 0


def _cmd_info() -> int:
    print("WorldQuant Alpha101（Kakushadze 2016, arxiv:1601.00991）")
    print(f"  共 101 个：仅量价 82 个；需行业分类 18 个"
          f"（{', '.join(sorted(ALPHA101_NEEDS_INDUSTRY))}）；")
    print(f"  需市值 cap 1 个（{', '.join(sorted(ALPHA101_NEEDS_CAP))}）。")
    print("Qlib Alpha158 量价部分（Microsoft Qlib, MIT License）")
    print(f"  共 158 个：K 线 9 + 原始价格 4 + 滚动算子 {len(ALPHA158_ROLLING)}"
          f" 种 × {len(ALPHA158_WINDOWS)} 窗口（{ALPHA158_WINDOWS}）。")
    print("输入列：open/high/low/close/volume/amount（vwap=amount/volume，"
          "returns=close.pct_change()）。")
    return 0


def _stream_compute(compute_fn, names: list[str], frames: dict[str, pd.DataFrame],
                    outdir: Path, latest_csv: str, summ_csv: str,
                    label: str) -> None:
    """逐因子计算并落盘（大股票池下避免同时持有全部因子矩阵）。

    每个因子单独调用 compute_fn(frames, include=[name])，取其统计量后
    立即释放；内存占用与因子数无关，与股票数×交易日数成正比。
    """
    t0 = time.time()
    summ_rows: list[dict] = []
    latest_cols: dict[str, pd.Series] = {}
    for i, name in enumerate(names):
        res = compute_fn(frames, include=[name])[name]
        vals = res.to_numpy(dtype=float)
        finite = np.isfinite(vals)
        summ_rows.append({
            "factor": name,
            "n_finite": int(finite.sum()),
            "coverage": float(finite.mean()),
            "mean": float(np.nanmean(vals)) if finite.any() else np.nan,
            "std": float(np.nanstd(vals)) if finite.any() else np.nan,
        })
        latest = np.full(vals.shape[1], np.nan)
        for j in range(vals.shape[1]):
            col_finite = np.flatnonzero(finite[:, j])
            if col_finite.size:
                latest[j] = vals[col_finite[-1], j]
        latest_cols[name] = pd.Series(latest, index=res.columns)
        del res, vals, finite
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[进度] {label}: {i + 1}/{len(names)}，用时 {el:.0f}s",
                  flush=True)
    pd.DataFrame(summ_rows).to_csv(outdir / summ_csv, index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(latest_cols).to_csv(outdir / latest_csv, encoding="utf-8-sig")
    cov = pd.Series({r["factor"]: r["coverage"] for r in summ_rows})
    print(f"[ok] {label}: {len(names)} 个，用时 {time.time() - t0:.0f}s；"
          f"有效值占比 中位 {cov.median():.1%}、最低 {cov.min():.1%}"
          f"（{cov.idxmin()}）")


def _cmd_compute(args: argparse.Namespace) -> int:
    datadir = args.datadir
    if args.pool is not None and not args.no_fetch:
        if args.source == "tsp" or args.pool == "full_a":
            import screener  # noqa: PLC0415 仅此路径需要
            universe = screener.fetch_universe_full_a(tsp_dir=args.tsp_dir,
                                                      offline=args.offline)
        else:
            import screener  # noqa: PLC0415
            universe = screener.fetch_universe(offline=args.offline)
        rc = screener.run_fetch(
            universe, args.start or "20180101", args.end or "20261231",
            datadir, args.sleep, args.retries, args.max_stocks, args.offline,
            source=args.source, tsp_dir=args.tsp_dir)
        if rc != 0:
            print("[warn] 部分股票拉取失败，仅对已有数据计算", file=sys.stderr)
        symbols = universe["symbol"].tolist()
        if args.max_stocks:
            symbols = symbols[:args.max_stocks]
    elif args.no_fetch or (args.pool is None and not args.symbols):
        # 复用 --datadir 下已有 CSV（--no-fetch；如其他工作目录先前导入的缓存）
        import re
        symbols = sorted({p.stem.split("_")[0]
                          for p in Path(datadir).glob(f"*_{args.adjust}.csv")
                          if re.fullmatch(r"\d{6}", p.stem.split("_")[0])})
        if args.max_stocks:
            symbols = symbols[:args.max_stocks]
        if not symbols:
            print(f"{datadir} 下没有任何 *_{args.adjust}.csv，无法 --no-fetch",
                  file=sys.stderr)
            return 1
    else:
        symbols = args.symbols
        if not symbols:
            print("请用 --symbols 指定股票代码，或 --pool 指定股票池",
                  file=sys.stderr)
            return 1

    print(f"[info] 加载 {len(symbols)} 只股票的日线（{datadir}）...")
    frames, _names = load_frames(symbols, datadir=datadir, adjust=args.adjust,
                                 start=args.start, end=args.end)
    n_days, n_stocks = frames["close"].shape
    print(f"[ok] 面板: {n_days} 个交易日 × {n_stocks} 只股票")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.only in ("101", "both"):
        _stream_compute(compute_alpha101, ALPHA101_NAMES, frames, outdir,
                        "alpha101_latest.csv", "alpha101_coverage_summary.csv",
                        "Alpha101")
    if args.only in ("158", "both"):
        _stream_compute(compute_alpha158, ALPHA158_NAMES, frames, outdir,
                        "alpha158_latest.csv", "alpha158_coverage_summary.csv",
                        "Alpha158")

    print(f"[ok] 输出目录: {outdir}/（alpha101_latest.csv / "
          f"alpha158_latest.csv / *_coverage_summary.csv）")
    print(DISCLAIMER)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="开源经典因子库：WorldQuant Alpha101 + Qlib Alpha158（量价部分）",
        epilog="示例: python scripts/alpha_factors.py compute "
               "--symbols 000001 600519 --start 20190101")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="合成数据自检（不访问网络）")
    sub.add_parser("info", help="打印因子清单与数据要求")

    p = sub.add_parser("compute", help="对本地日线计算因子并输出最新一期横截面")
    p.add_argument("--symbols", nargs="*", default=[],
                   help="股票代码列表（6 位数字），如 000001 600519")
    p.add_argument("--pool", choices=["cs800", "full_a"], default=None,
                   help="股票池（cs800=沪深300+中证500；full_a=全 A）；"
                        "full_a 默认从本机 TSP parquet 导入（--source tsp）")
    p.add_argument("--source", choices=["em", "tx", "tsp"], default="tsp",
                   help="行情来源（tsp=本机 TSP parquet，默认；em/tx=akshare）")
    p.add_argument("--tsp-dir", default=r"D:\tsp-data",
                   help="本机 TSP 数据目录（--source tsp 时使用）")
    p.add_argument("--max-stocks", type=int, default=None,
                   help="只对股票池前 N 只计算（调试/抽样用）")
    p.add_argument("--start", default=None, help="开始日期 YYYYMMDD")
    p.add_argument("--end", default=None, help="结束日期 YYYYMMDD")
    p.add_argument("--datadir", default="data/daily", help="日线 CSV 目录")
    p.add_argument("--adjust", default="qfq", help="复权方式（qfq/hfq/none）")
    p.add_argument("--outdir", default="output/factors", help="输出目录")
    p.add_argument("--only", choices=["101", "158", "both"], default="both",
                   help="只计算哪个因子库")
    p.add_argument("--offline", action="store_true",
                   help="offline 模式（只用本地缓存，不访问网络）")
    p.add_argument("--no-fetch", action="store_true",
                   help="不拉取/导入数据：直接使用 --datadir 下已有的 "
                        "<代码>_<复权>.csv（复用先前导入的缓存）")
    p.add_argument("--sleep", type=float, default=0.3, help="批量拉取间隔（秒）")
    p.add_argument("--retries", type=int, default=3, help="单只股票重试次数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "selftest":
        return _selftest()
    if args.command == "info":
        return _cmd_info()
    return _cmd_compute(args)


if __name__ == "__main__":
    sys.exit(main())
