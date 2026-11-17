"""可插拔策略接口与内置经典策略（量化学习项目第 3 步）。

设计约定
--------
1. 每个策略继承 :class:`Strategy`，只负责两件事：
   - **参数注册**：类属性 ``params`` 声明参数名、默认值、类型与取值范围，
     构造时统一做类型转换与校验，未注册的参数直接报错；
   - **信号生成**：``positions(df)`` 返回长度与 df 相同的目标仓位数组，
     ``positions[t]`` ∈ {0, 1} 表示 t 日收盘后应持有的目标仓位
     （1=全仓做多，0=空仓持现金）。
2. 无未来函数：``positions[t]`` 只能使用 t 日及之前的信息；
   回测引擎在 t+1 日以开盘价按仓位**变化（边沿）**成交，见 scripts/backtest.py。
3. 数据未就绪（如均线热身期）的位置用 NaN 表示，引擎不会据此交易。
4. 每个策略必须在 ``sources`` 中注明参考来源链接。本项目全部内容仅为
   学习用途：历史模拟回测结果不代表未来表现，不构成投资建议。

内置策略一览（``name`` → 说明）：
- ``ma_cross``：双均线金叉/死叉（第 2 步策略原样迁移，验证接口等价性）；
- ``rsi``：RSI 超卖反转（Wilder 1978 原始平滑定义）；
- ``bollinger``：布林带均值回归（跌破下轨回升入场、回到中轨离场）；
- ``momentum``：绝对动量 / 动量轮动到现金（Jegadeesh-Titman 动量 +
  Antonacci 绝对动量腿，单标的框架下的经典退化形式）。

开源实现参考（学习对照，未直接引用代码）：
- backtrader（GPLv3）指标与策略框架: https://github.com/cloudquant/backtrader
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    """一个已注册参数的元信息：默认值、类型、取值范围与说明。"""

    default: Any
    kind: type = float
    min: float | None = None
    max: float | None = None
    help: str = ""


class Strategy(abc.ABC):
    """策略基类：参数注册 + 信号生成。

    子类必须定义：``name``（命令行选择用）、``params``（参数注册表）、
    ``sources``（参考来源链接列表）、``positions``（信号生成方法）。
    可选覆盖：``warmup``（生成首个有效信号所需的最少交易日数）、
    ``validate``（跨参数校验，如要求 short < long）。
    """

    name: str = "base"
    params: dict[str, ParamSpec] = {}
    sources: list[str] = []

    def __init__(self, **overrides: Any) -> None:
        unknown = sorted(set(overrides) - set(self.params))
        if unknown:
            raise ValueError(
                f"策略 {self.name} 未注册参数: {', '.join(unknown)}；"
                f"已注册参数: {', '.join(self.params)}"
            )
        self._values: dict[str, Any] = {}
        for pname, spec in self.params.items():
            raw = overrides.get(pname, spec.default)
            try:
                if spec.kind is bool and isinstance(raw, str):
                    # CLI 传入的字符串布尔值做显式解析（"False" 不能按真值转 True）
                    value = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = spec.kind(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"参数 {pname}={raw!r} 无法转换为 {spec.kind.__name__}"
                ) from None
            if isinstance(value, (int, float)):
                if spec.min is not None and value < spec.min:
                    raise ValueError(f"参数 {pname}={value} 小于下限 {spec.min}")
                if spec.max is not None and value > spec.max:
                    raise ValueError(f"参数 {pname}={value} 大于上限 {spec.max}")
            self._values[pname] = value
        self.validate()

    def validate(self) -> None:
        """跨参数校验钩子，子类可覆盖。"""

    def get(self, name: str) -> Any:
        """读取已转换、已校验的参数值。"""
        return self._values[name]

    @property
    def warmup(self) -> int:
        """生成第一个有效信号所需的最少交易日数（数据预热用，默认 0）。"""
        return 0

    def param_string(self) -> str:
        """参数串（k=v,k=v），用于统一报告与文件名。"""
        return ",".join(f"{k}={self._values[k]}" for k in self.params)

    def label(self) -> str:
        """策略可读标签，用于统一报告与图表标题，如 ``ma_cross(short=5,long=20)``。"""
        return f"{self.name}({self.param_string()})"

    def file_label(self) -> str:
        """文件名友好的紧凑标签，如 ``ma_cross_5_20`` / ``rsi_14_30.0_70.0``。"""
        return self.name + "_" + "_".join(str(self._values[k]) for k in self.params)

    def describe(self) -> dict[str, Any]:
        """统一报告用的策略元信息。"""
        return {"name": self.name, "label": self.label(), "params": dict(self._values)}

    @abc.abstractmethod
    def positions(self, df: pd.DataFrame) -> np.ndarray:
        """返回目标仓位数组（0/1，NaN=数据未就绪），只允许使用当日及之前的信息。"""


# ---------------------------------------------------------------------------
# 双均线（第 2 步策略的原样迁移）
# ---------------------------------------------------------------------------

class MACrossStrategy(Strategy):
    """双均线金叉做多/死叉清仓——本项目第 2 步策略迁移到统一接口。

    规则：短期均线上穿长期均线 → 目标仓位 1（做多）；
          短期均线下穿长期均线 → 目标仓位 0（空仓持现金）。
    """

    name = "ma_cross"
    params = {
        "short": ParamSpec(5, int, min=1, max=1000, help="短期均线窗口（交易日）"),
        "long": ParamSpec(20, int, min=2, max=2000, help="长期均线窗口（交易日）"),
    }
    sources = [
        "本项目第 2 步自研双均线框架：docs/quant-platform-research.md 第 9 节",
    ]

    def validate(self) -> None:
        short, long = self._values["short"], self._values["long"]
        if short >= long:
            raise ValueError(f"短期窗口({short})必须小于长期窗口({long})")

    @property
    def warmup(self) -> int:
        return int(self._values["long"])

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        short, long = int(self._values["short"]), int(self._values["long"])
        ma_s = pd.Series(close).rolling(short).mean().values
        ma_l = pd.Series(close).rolling(long).mean().values
        pos = np.where(~np.isnan(ma_s) & ~np.isnan(ma_l), (ma_s > ma_l).astype(float), np.nan)
        return pos


# ---------------------------------------------------------------------------
# RSI 超卖反转
# ---------------------------------------------------------------------------

def wilder_rsi(close: pd.Series, period: int) -> np.ndarray:
    """Wilder (1978) 原始定义的 RSI：涨跌幅用 Wilder 平滑（alpha=1/period）累加。

    RSI = 100 × avg_gain / (avg_gain + avg_loss)，avg_loss=0（纯上涨）时为 100。
    """

    delta = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    denom = avg_gain + avg_loss
    rsi = np.where(denom.values == 0, 100.0, 100.0 * avg_gain.values / denom.values)
    rsi[: max(period - 1, 0)] = np.nan  # 前 period-1 天数据未就绪
    return rsi


class RSIReversalStrategy(Strategy):
    """RSI 超卖反转：从超卖区向上穿越入场，RSI 自 exit_level 上方跌回后离场。

    规则（经典 30/70 用法）：
    - RSI 从超卖区（≤ oversold）向上穿越 oversold → 目标仓位 1（超卖反转做多）；
    - 持仓期间，RSI 从 exit_level 上方跌回其下方 → 目标仓位 0（清仓）。
    说明：RSI 采用 Wilder 原始平滑口径；exit_level=70 即经典"到超买区才离场"，
    exit_level=50 则是"回归中轴离场"的均值回归变体，两者都可在参数扫描中比较。
    """

    name = "rsi"
    params = {
        "period": ParamSpec(14, int, min=2, max=500, help="RSI 计算窗口（Wilder 经典 14）"),
        "oversold": ParamSpec(30.0, float, min=0.0, max=50.0, help="超卖阈值（经典 30）"),
        "exit_level": ParamSpec(70.0, float, min=1.0, max=99.0, help="离场阈值（经典 70；50 为回归中轴离场）"),
    }
    sources = [
        "J. Welles Wilder, New Concepts in Technical Trading Systems (Trend Research, 1978)",
        "StockCharts ChartSchool: Relative Strength Index (RSI) — "
        "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/"
        "technical-indicators/relative-strength-index-rsi",
    ]

    def validate(self) -> None:
        if self._values["exit_level"] <= self._values["oversold"]:
            raise ValueError(f"exit_level({self._values['exit_level']})必须大于 oversold({self._values['oversold']})")

    @property
    def warmup(self) -> int:
        return int(self._values["period"]) + 1

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        rsi = wilder_rsi(df["close"], int(self._values["period"]))
        oversold, exit_level = self._values["oversold"], self._values["exit_level"]
        ready = ~np.isnan(rsi)
        pos = np.full(len(df), np.nan)
        state = 0.0
        for t in range(len(df)):
            if not ready[t]:
                continue
            if t > 0 and ready[t - 1]:
                prev, cur = rsi[t - 1], rsi[t]
                if state == 0 and prev <= oversold < cur:
                    state = 1.0  # 超卖区向上穿越 → 反转做多
                elif state == 1 and prev >= exit_level > cur:
                    state = 0.0  # 自 exit_level 上方跌回 → 清仓
            pos[t] = state
        return pos


# ---------------------------------------------------------------------------
# 布林带均值回归
# ---------------------------------------------------------------------------

class BollingerReversionStrategy(Strategy):
    """布林带均值回归：跌破下轨后回升入场，回到中轨上方离场；可选趋势过滤。

    规则（经典"回归"用法）：
    - 收盘价从下轨下方回升至下轨上方 → 目标仓位 1（超跌反弹做多）；
    - 收盘价从中轨下方回升至中轨上方 → 目标仓位 0（均值回归完成离场）；
    - 可选趋势过滤：trend_period > 0 时，仅当收盘价在 trend_period 日均线上方
      才允许入场（不在下跌趋势中接飞刀）。
    布林带定义：中轨 = N 日收盘均值，上下轨 = 中轨 ± num_std × N 日标准差
    （John Bollinger 原书口径使用总体标准差）。
    """

    name = "bollinger"
    params = {
        "period": ParamSpec(20, int, min=2, max=500, help="布林带窗口（经典 20）"),
        "num_std": ParamSpec(2.0, float, min=0.1, max=10.0, help="带宽倍数（经典 2.0）"),
        "trend_period": ParamSpec(0, int, min=0, max=500, help="趋势过滤均线窗口（0=关闭；>0 时只在均线上方入场）"),
    }
    sources = [
        "John Bollinger, Bollinger on Bollinger Bands (McGraw-Hill, 2001)",
        "Bollinger 官方站点规则说明: https://www.bollingerbands.com/bollinger-bands",
    ]

    @property
    def warmup(self) -> int:
        return max(int(self._values["period"]), int(self._values["trend_period"]))

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        period, k = int(self._values["period"]), float(self._values["num_std"])
        trend_n = int(self._values["trend_period"])
        mid = close.rolling(period).mean()
        lower = mid - k * close.rolling(period).std(ddof=0)
        trend_ma = close.rolling(trend_n).mean() if trend_n > 0 else None
        ready = ~mid.isna().values
        pos = np.full(len(df), np.nan)
        state = 0.0
        for t in range(len(df)):
            if not ready[t]:
                continue
            trend_ok = trend_ma is None or (
                not np.isnan(trend_ma.iloc[t]) and close.iloc[t] > trend_ma.iloc[t]
            )
            if t > 0 and ready[t - 1]:
                pc, cc = close.iloc[t - 1], close.iloc[t]
                if state == 0 and pc <= lower.iloc[t - 1] < cc and trend_ok:
                    state = 1.0  # 自下轨下方回升 → 超跌反弹做多（可选趋势过滤）
                elif state == 1 and pc <= mid.iloc[t - 1] < cc:
                    state = 0.0  # 回到中轨上方 → 均值回归完成
            pos[t] = state
        return pos


# ---------------------------------------------------------------------------
# 动量（绝对动量 / 动量轮动到现金）
# ---------------------------------------------------------------------------

class MomentumStrategy(Strategy):
    """绝对动量：动量超过阈值则持仓，否则空仓持现金（资产与现金之间的轮动）。

    规则：
    - t 日信号用过去 ``lookback`` 个交易日的累计收益率衡量动量，
      并跳过最近 ``skip`` 个交易日（规避短期反转，Jegadeesh-Titman 1993 的做法）；
    - 动量 > threshold → 目标仓位 1；否则 → 0。
    单标的框架下，"横截面动量轮动"退化为**绝对动量**：在"该资产"与"现金"
    之间轮动，只有动量跑赢现金（threshold=0 时即动量 > 0）才持有。
    经典参数参考：J-T 用 12-1 月动量（约 252 日回看、跳过 21 日）；
    Antonacci 绝对动量腿用 12 个月总收益与无风险利率比较。
    """

    name = "momentum"
    params = {
        "lookback": ParamSpec(120, int, min=10, max=2000, help="动量回看窗口（交易日；J-T 经典约 252）"),
        "skip": ParamSpec(5, int, min=0, max=100, help="跳过的最近交易日数（J-T 经典约 21）"),
        "threshold": ParamSpec(0.0, float, min=-1.0, max=1.0, help="建仓动量阈值（默认 0：跑赢现金即持有）"),
    }
    sources = [
        "Jegadeesh, N. & Titman, S. (1993), 'Returns to Buying Winners and Selling Losers: "
        "Implications for Stock Market Efficiency', Journal of Finance 48(1), 65-91",
        "Wharton-Jacobs Levy 奖页面（J-T 动量论文获奖）: "
        "https://drupal.prod.pionline.com/awards/narasimhan-jegadeesh-and-sheridan-titman-win-wharton-jacobs-levy-prize",
        "Antonacci, G., 'Risk Premia Harvesting Through Dual Momentum'（绝对动量腿）: "
        "https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2042750",
    ]

    @property
    def warmup(self) -> int:
        return int(self._values["lookback"]) + int(self._values["skip"]) + 1

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        lb, sk = int(self._values["lookback"]), int(self._values["skip"])
        thr = float(self._values["threshold"])
        pos = np.full(len(df), np.nan)
        for t in range(len(df)):
            j = t - sk - 1  # 动量终点：跳过最近 sk 天
            i = j - lb + 1  # 动量起点：再往前 lb 天
            if i < 0:
                continue
            mom = close[j] / close[i] - 1.0
            pos[t] = 1.0 if mom > thr else 0.0
        return pos


# ---------------------------------------------------------------------------
# 新低反转（TSP 开源策略种子 n_day_low_reversal 的移植）
# ---------------------------------------------------------------------------

class NDayLowReversalStrategy(Strategy):
    """新低反转：触及 N 日收盘新低且当日收阳放量 → 做多；跌破均线/止损/到期离场。

    移植自 TSP 平台内置策略种子 ``n_day_low_reversal``（"触及60日新低后当日
    收阳放量, 反转信号"），本仓库适配要点（TSP 原语义见种子文件注释）：
    - **入场**（t 日收盘，全部满足才触发）：
      1. 触及 N 日收盘新低：``close[t] <= min(close[t-N+1..t])``
         （TSP 的 low_60d 是**收盘价**的 60 日滚动最小值，含当日）；
      2. 收阳：``close[t] > open[t]``；
      3. 量比过滤：``volume[t] / mean(volume[t-5..t-1]) >= vol_ratio_min``
         （TSP vol_ratio_5d 同口径：前 5 日均量，不含当日）；
    - **离场**（任一触发）：
      1. 均线跌破：收盘自 exit_ma 均线上方跌至其下方（TSP 的
         signal_ma20_breakdown，默认 20 日均线）；
      2. 止损：收盘 <= 入场参考价 × (1 + stop_loss)（入场参考价取 t+1 日
         开盘——本仓库引擎 t 日信号 t+1 开盘成交；默认 -6%）；
      3. 最长持有到期（默认 15 个交易日）。
    - 引擎口径与其余策略一致：t 日收盘信号在 t+1 日开盘成交，止损以收盘价
      判定、下一开盘价成交（跳空风险如实保留）。

    参考来源：
    - TSP 平台内置种子（本机落盘副本）：
      D:\\tsp-data\\strategies\\custom\\n_day_low_reversal.py；
    - TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel
    """

    name = "n_day_low_reversal"
    params = {
        "n_days": ParamSpec(60, int, min=2, max=1000, help="新低窗口（交易日，TSP 种子默认 60）"),
        "require_n_day_low": ParamSpec(True, bool, help="要求触及 N 日收盘新低"),
        "require_bullish_candle": ParamSpec(True, bool, help="要求当日收阳（收盘>开盘）"),
        "use_volume_filter": ParamSpec(True, bool, help="启用量比过滤"),
        "vol_ratio_min": ParamSpec(1.5, float, min=0.0, max=50.0, help="最低量比（当日量/前5日均量）"),
        "stop_loss": ParamSpec(-0.06, float, min=-0.5, max=0.0, help="止损线（相对入场价，负值；0=关闭）"),
        "max_hold_days": ParamSpec(15, int, min=0, max=500, help="最长持有交易日（0=关闭）"),
        "exit_ma": ParamSpec(20, int, min=2, max=1000, help="离场均线窗口（收盘跌破离场）"),
    }
    sources = [
        "TSP 平台内置策略种子 n_day_low_reversal（本机 D:\\tsp-data\\strategies\\custom 落盘副本）",
        "TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel",
    ]

    @property
    def warmup(self) -> int:
        return max(int(self._values["n_days"]), int(self._values["exit_ma"]), 6)

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        open_ = df["open"].values
        volume = (df["volume"].values if "volume" in df.columns
                  else np.full(len(df), np.nan))
        n = len(df)
        n_days = int(self._values["n_days"])
        exit_ma = int(self._values["exit_ma"])
        vol_min = float(self._values["vol_ratio_min"])
        stop = float(self._values["stop_loss"])
        max_hold = int(self._values["max_hold_days"])
        require_low = bool(self._values["require_n_day_low"])
        require_bull = bool(self._values["require_bullish_candle"])
        use_vol = bool(self._values["use_volume_filter"])

        # 新低 = 收盘价 N 日滚动最小值（含当日，TSP low_60d 同口径）
        low_nd = pd.Series(close).rolling(n_days).min().values
        ma_e = pd.Series(close).rolling(exit_ma).mean().values
        # 量比 = 当日成交量 / 前 5 日均量（不含当日，TSP vol_ratio_5d 同口径）
        vol_prev_mean = pd.Series(volume).rolling(5).mean().shift(1).values
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = volume / vol_prev_mean

        ready = ~np.isnan(low_nd) & ~np.isnan(ma_e)
        pos = np.full(n, np.nan)
        state = 0.0
        entry_t = -1
        entry_ref = np.nan  # 入场参考价 = 信号次日开盘（止损基准）
        for t in range(n):
            if not ready[t]:
                continue
            if state == 0.0:
                cond = True
                if require_low:
                    cond = cond and close[t] <= low_nd[t]
                if require_bull:
                    cond = cond and close[t] > open_[t]
                if use_vol:
                    cond = cond and np.isfinite(vol_ratio[t]) and \
                        vol_ratio[t] >= vol_min
                if cond and np.isfinite(open_[t]):
                    state = 1.0
                    entry_t = t
                    # 引擎在 t+1 开盘成交；末日无次日开盘时退化为 t 日收盘
                    entry_ref = open_[t + 1] if (
                        t + 1 < n and np.isfinite(open_[t + 1])) else close[t]
            else:
                ma_break = (t > 0 and ready[t - 1]
                            and close[t] < ma_e[t] and close[t - 1] >= ma_e[t - 1])
                stop_hit = stop < 0.0 and close[t] <= entry_ref * (1.0 + stop)
                hold_hit = max_hold > 0 and t - entry_t >= max_hold
                if ma_break or stop_hit or hold_hit:
                    state = 0.0
                    entry_t = -1
                    entry_ref = np.nan
            pos[t] = state
        return pos


# ---------------------------------------------------------------------------
# 超跌反转（TSP 开源策略种子 oversold_volume_reversal 的移植）
# ---------------------------------------------------------------------------

class OversoldVolumeReversalStrategy(Strategy):
    """超跌反转：深跌后放量阳线确认的反转买点；5 日动量转正离场。

    移植自 TSP 平台策略种子 ``oversold_volume_reversal``（"超跌放量反转，
    新低反转的增强版（自主设计 v1）"），本仓库适配要点（TSP 原语义见种子
    文件注释）：
    - **入场**（t 日收盘，全部满足才触发）：
      1. 深跌：``close[t-1] / close[t-lookback] - 1 <= max_return``
         （TSP 同口径：昨收相对 lookback 日前收盘的区间收益；max_return
         以百分数注册，内部 /100）；
      2. 量比：``volume[t] / mean(volume[t-5..t-1]) >= vol_ratio_min``
         （TSP vol_ratio_5d 同口径：前 5 日均量，不含当日）；
      3. 阳线：``close[t] > close[t-1]``（TSP require_bullish 同口径，
         收盘高于昨收）。
    - **离场**（任一触发）：
      1. 5 日动量转正：``close[t] / close[t-5] - 1 > 0``（TSP 的
         signal_ovr_done，"反转完成退出"）；
      2. 止损：收盘 <= 入场参考价 × (1 + stop_loss)（入场参考价取 t+1 日
         开盘——本仓库引擎 t 日信号 t+1 开盘成交；默认 -8%）；
      3. 最长持有到期（默认 20 个交易日）。
    - 引擎口径与其余策略一致：t 日收盘信号在 t+1 日开盘成交，止损以收盘价
      判定、下一开盘价成交（跳空风险如实保留）。

    参考来源：
    - TSP 平台自定义策略种子（本机落盘副本）：
      D:\\tsp-data\\strategies\\custom\\oversold_volume_reversal.py；
    - TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel
    """

    name = "oversold_volume_reversal"
    params = {
        "lookback": ParamSpec(15, int, min=2, max=200, help="深跌回看窗口（交易日，TSP 种子默认 15）"),
        "max_return": ParamSpec(-30.0, float, min=-95.0, max=0.0, help="区间收益上限（%，默认 -30：15 日跌超 30% 才入场）"),
        "vol_ratio_min": ParamSpec(1.5, float, min=0.0, max=50.0, help="最低量比（当日量/前5日均量）"),
        "require_bullish": ParamSpec(True, bool, help="要求当日阳线（收盘>昨收）"),
        "stop_loss": ParamSpec(-0.08, float, min=-0.5, max=0.0, help="止损线（相对入场价，负值；0=关闭）"),
        "max_hold_days": ParamSpec(20, int, min=0, max=500, help="最长持有交易日（0=关闭）"),
    }
    sources = [
        "TSP 平台自定义策略种子 oversold_volume_reversal（本机 D:\\tsp-data\\strategies\\custom 落盘副本）",
        "TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel",
    ]

    @property
    def warmup(self) -> int:
        return int(self._values["lookback"]) + 30  # TSP required_warmup_bars 同口径

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        open_ = df["open"].values
        volume = (df["volume"].values if "volume" in df.columns
                  else np.full(len(df), np.nan))
        n = len(df)
        lookback = int(self._values["lookback"])
        max_ret = float(self._values["max_return"]) / 100.0
        vol_min = float(self._values["vol_ratio_min"])
        stop = float(self._values["stop_loss"])
        max_hold = int(self._values["max_hold_days"])
        require_bull = bool(self._values["require_bullish"])

        # 区间收益 = 昨收 / lookback 日前收盘 - 1（TSP 同口径）
        past = pd.Series(close).shift(1).values
        base = pd.Series(close).shift(lookback).values
        with np.errstate(divide="ignore", invalid="ignore"):
            window_ret = np.where(base > 0, past / base - 1.0, np.nan)
        # 量比 = 当日成交量 / 前 5 日均量（不含当日，TSP vol_ratio_5d 同口径）
        vol_prev_mean = pd.Series(volume).rolling(5).mean().shift(1).values
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = volume / vol_prev_mean
        # 5 日动量（TSP momentum_5d 同口径：5 日前收盘为基）
        mom5 = pd.Series(close).shift(5).values
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum_5d = np.where(mom5 > 0, close / mom5 - 1.0, np.nan)

        pos = np.full(n, np.nan)
        state = 0.0
        entry_t = -1
        entry_ref = np.nan  # 入场参考价 = 信号次日开盘（止损基准）
        for t in range(n):
            ready = (t >= lookback and np.isfinite(close[t])
                     and np.isfinite(window_ret[t]))
            if not ready:
                continue
            if state == 0.0:
                cond = window_ret[t] <= max_ret
                cond = cond and np.isfinite(vol_ratio[t]) and vol_ratio[t] >= vol_min
                if require_bull:
                    cond = cond and t > 0 and close[t] > close[t - 1]
                if cond and np.isfinite(open_[t]):
                    state = 1.0
                    entry_t = t
                    # 引擎在 t+1 开盘成交；末日无次日开盘时退化为 t 日收盘
                    entry_ref = open_[t + 1] if (
                        t + 1 < n and np.isfinite(open_[t + 1])) else close[t]
            else:
                mom_done = (np.isfinite(momentum_5d[t]) and momentum_5d[t] > 0.0)
                stop_hit = stop < 0.0 and close[t] <= entry_ref * (1.0 + stop)
                hold_hit = max_hold > 0 and t - entry_t >= max_hold
                if mom_done or stop_hit or hold_hit:
                    state = 0.0
                    entry_t = -1
                    entry_ref = np.nan
            pos[t] = state
        return pos


# ---------------------------------------------------------------------------
# 单吊王通道（TSP 开源策略种子 dandiao_channel 的移植）
# ---------------------------------------------------------------------------

class DandiaoChannelStrategy(Strategy):
    """单吊王通道：布林 + MACD + 唐奇安通道多指标分级共振，≥N 指标确认才买。

    移植自 TSP 平台策略种子 ``dandiao_channel``（"单吊王通道 — 多指标分级
    共振，半岛路单吊王 11.16 视频拆解的可回测版"），本仓库适配要点：
    - **入场**（t 日收盘，三个信号合计 >= confirm 才触发）：
      1. 布林带回升：``close[t-1] <= boll_lower[t-1] 且 close[t] >
         boll_lower[t]``（布林 20 日、2.0 倍总体标准差，TSP matrix_feature
         默认口径）；
      2. MACD 金叉：DIF（EMA12-EMA26）自 0 轴下方上穿 0 轴
         （``dif[t-1] <= 0 且 dif[t] > 0``，标准 MACD 12/26 口径）；
      3. 唐奇安通道下轨支撑：``close[t-1] <= don_lo[t-1] 且 close[t] >
         don_lo[t]``（don_lo = chan 日最低价滚动最小值，含当日）。
    - **离场**（任一触发）：止损（默认 -8%）与最长持有到期（默认 20 个
      交易日）；TSP 种子无主动离场信号（EXIT_SIGNALS 为空），本仓库照搬。
    - 引擎口径与其余策略一致：t 日收盘信号在 t+1 日开盘成交，止损以收盘价
      判定、下一开盘价成交（跳空风险如实保留）。

    参考来源：
    - TSP 平台自定义策略种子（本机落盘副本）：
      D:\\tsp-data\\strategies\\custom\\dandiao_channel.py；
    - TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel
    """

    name = "dandiao_channel"
    params = {
        "confirm": ParamSpec(2, int, min=1, max=3, help="共振确认数（≥N 个信号才买；TSP 全期回测最优为 3）"),
        "chan": ParamSpec(20, int, min=5, max=120, help="唐奇安通道天数（TSP 种子默认 20）"),
        "stop_loss": ParamSpec(-0.08, float, min=-0.5, max=0.0, help="止损线（相对入场价，负值；0=关闭）"),
        "max_hold_days": ParamSpec(20, int, min=0, max=500, help="最长持有交易日（0=关闭）"),
    }
    sources = [
        "TSP 平台自定义策略种子 dandiao_channel（本机 D:\\tsp-data\\strategies\\custom 落盘副本）",
        "TSP 项目仓库: https://github.com/chenjifengtrading/tick-stock-panel",
    ]

    @property
    def warmup(self) -> int:
        # TSP required_warmup_bars=60；布林 20 / MACD 26 / 唐奇安 chan 全覆盖
        return max(int(self._values["chan"]), 60)

    def positions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values
        open_ = df["open"].values
        low = (df["low"].values if "low" in df.columns
               else np.full(len(df), np.nan))
        n = len(df)
        confirm = int(self._values["confirm"])
        chan = int(self._values["chan"])
        stop = float(self._values["stop_loss"])
        max_hold = int(self._values["max_hold_days"])

        # 布林带 20 日 / 2.0 倍总体标准差（TSP matrix_feature 默认口径）
        mid = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std(ddof=0).values
        boll_lower = mid - 2.0 * std
        # MACD DIF = EMA12 - EMA26（标准 MACD 口径）
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        # 唐奇安通道：chan 日最低价滚动最小值（含当日）
        don_lo = pd.Series(low).rolling(chan).min().values

        # 信号1: 布林带下轨回升 / 信号2: DIF 上穿 0 轴 / 信号3: 通道下轨支撑
        pc = pd.Series(close).shift(1).values
        pd_ = pd.Series(dif).shift(1).values
        pl = pd.Series(boll_lower).shift(1).values
        pdon = pd.Series(don_lo).shift(1).values
        boll_buy = (pc <= pl) & (close > boll_lower)
        macd_buy = (pd_ <= 0) & (dif > 0)
        ch_buy = (pc <= pdon) & (close > don_lo)
        buy_score = boll_buy.astype(int) + macd_buy.astype(int) + ch_buy.astype(int)

        ready = ~np.isnan(boll_lower) & ~np.isnan(dif) & ~np.isnan(don_lo)
        pos = np.full(n, np.nan)
        state = 0.0
        entry_t = -1
        entry_ref = np.nan  # 入场参考价 = 信号次日开盘（止损基准）
        for t in range(n):
            if not ready[t]:
                continue
            if state == 0.0:
                if buy_score[t] >= confirm and np.isfinite(open_[t]):
                    state = 1.0
                    entry_t = t
                    # 引擎在 t+1 开盘成交；末日无次日开盘时退化为 t 日收盘
                    entry_ref = open_[t + 1] if (
                        t + 1 < n and np.isfinite(open_[t + 1])) else close[t]
            else:
                stop_hit = stop < 0.0 and close[t] <= entry_ref * (1.0 + stop)
                hold_hit = max_hold > 0 and t - entry_t >= max_hold
                if stop_hit or hold_hit:
                    state = 0.0
                    entry_t = -1
                    entry_ref = np.nan
            pos[t] = state
        return pos


# ---------------------------------------------------------------------------
# 策略注册表
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (MACrossStrategy, RSIReversalStrategy, BollingerReversionStrategy,
                MomentumStrategy, NDayLowReversalStrategy,
                OversoldVolumeReversalStrategy, DandiaoChannelStrategy)
}

ALIASES = {"ma": "ma_cross"}


def list_strategies() -> str:
    """注册表说明文字（命令行帮助用）。"""
    lines = []
    for name, cls in REGISTRY.items():
        params = ", ".join(f"{k}={spec.default}" for k, spec in cls.params.items())
        lines.append(f"  {name}: {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''} (默认 {params})")
    return "\n".join(lines)


def create_strategy(name: str, **params: Any) -> Strategy:
    """按注册名创建策略实例；未注册参数/非法值会抛出带说明的 ValueError。"""
    key = ALIASES.get(name, name)
    cls = REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"未知策略 {name!r}，可选: {', '.join(sorted(REGISTRY))}")
    return cls(**params)
