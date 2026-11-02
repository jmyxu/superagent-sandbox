"""跨策略对比图：叠加各策略收益曲线与回撤（量化学习项目第 3 步）。

读取 scripts/backtest.py 统一报告产出的每日净值 CSV
（output/backtest/<代码>_<策略标签>_equity.csv），把多个策略的净值曲线、
买入持有与沪深300基准画在一起，下子图为各策略回撤。

用法示例：
    python scripts/plot_compare.py 000001 600519

默认对比默认参数的四个内置策略：
    ma_cross_5_20 / rsi_14_30.0_70.0 / bollinger_20_2.0_0 / momentum_120_5_0.0
可用 --labels 覆盖策略文件标签（空格分隔）。

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import DISCLAIMER, _setup_cjk_font  # noqa: E402

DEFAULT_LABELS = ["ma_cross_5_20", "rsi_14_30.0_70.0", "bollinger_20_2.0_0", "momentum_120_5_0.0"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="叠加各策略收益曲线与回撤（读取统一报告产出的每日净值 CSV）",
        epilog="示例: python scripts/plot_compare.py 000001 600519",
    )
    parser.add_argument("symbols", nargs="+", help="股票代码，可传多个")
    parser.add_argument("--outdir", default="output/backtest", help="回测输出目录，默认 output/backtest")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help=f"策略文件标签（默认: {' '.join(DEFAULT_LABELS)}）")
    parser.add_argument("--period", default="20200101 ~ 20251231", help="图表标题中的回测区间说明")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    args = parse_args(argv)
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    for raw in args.symbols:
        symbol = re.search(r"\d{6}", raw.strip()).group(0)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        colors = ["tab:blue", "tab:red", "tab:purple", "tab:olive"]
        missing: list[str] = []
        for label, color in zip(args.labels, colors):
            path = outdir / f"{symbol}_{label}_equity.csv"
            if not path.exists():
                missing.append(label)
                continue
            eq = pd.read_csv(path)
            dates = pd.to_datetime(eq["date"])
            ax1.plot(dates, eq["strategy_equity"], label=label, lw=1.2, color=color)
            dd = eq["strategy_equity"] / eq["strategy_equity"].cummax() - 1
            ax2.plot(dates, dd, lw=0.8, color=color, alpha=0.7)
        if missing:
            print(f"[warn] {symbol}: 缺少净值 CSV: {', '.join(missing)}", file=sys.stderr)
        if not any((outdir / f"{symbol}_{l}_equity.csv").exists() for l in args.labels):
            print(f"[failed] {symbol}: 没有任何策略净值 CSV 可用", file=sys.stderr)
            plt.close(fig)
            continue
        # 买入持有与基准取任意一个 CSV（统一报告格式相同）
        ref = pd.read_csv(outdir / f"{symbol}_{args.labels[0]}_equity.csv")
        dates = pd.to_datetime(ref["date"])
        ax1.plot(dates, ref["buy_hold"], label="买入持有（同期）", lw=1.0, alpha=0.6, color="tab:orange")
        ax1.plot(dates, ref["benchmark_300"], label="沪深300基准", lw=1.0, alpha=0.6, color="tab:green")
        ax1.set_title(f"{symbol} · 跨策略对比（{args.period}）")
        ax1.set_ylabel("净值（期初=1）")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)
        ax2.set_ylabel("策略回撤")
        ax2.grid(alpha=0.3)

        fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        out_png = outdir / f"{symbol}_strategy_compare.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[ok] 跨策略对比图已保存: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
