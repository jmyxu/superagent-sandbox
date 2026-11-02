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
# 策略注册表
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (MACrossStrategy, RSIReversalStrategy, BollingerReversionStrategy, MomentumStrategy)
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
