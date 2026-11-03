"""参数网格扫描与 walk-forward 样本外验证（量化学习项目第 4 步）。

解决的问题
----------
第 3 步的参数扫描全部在同一段历史上（2020-2025）事后挑选最优参数，
天然存在过拟合风险：样本内最优换个区间大概率重排。本步把整段历史拆成
3 个锚定式 walk-forward fold——每个 fold 只在样本内选参、把选出的参数
原样放到随后的样本外年份验证——检验"样本内最优"在样本外的存活率。

网格定义（每策略给出参数范围与步长，全部落在注册参数的合法区间内）：
  ma_cross : short ∈ {3,5,10,15,20}（范围 3–20，步长 5）
             long  ∈ {20,30,60,120,250}（范围 20–250）；剔除 short>=long
  rsi      : period ∈ {7,14,21,28}（范围 7–28，步长 7）
             oversold ∈ {20,25,30,35}（范围 20–35，步长 5）
             exit_level ∈ {50,60,70}（范围 50–70，步长 10）
  bollinger: period ∈ {10,20,50}（范围 10–50，经典 20）
             num_std ∈ {1.5,2.0,2.5,3.0}（范围 1.5–3.0，步长 0.5）
             trend_period ∈ {0,60,120}（范围 0–120，步长 60；0=关闭过滤）
  momentum : lookback ∈ {60,120,250}（范围 60–250，J-T 经典约 252）
             skip ∈ {0,5,21}（范围 0–21，J-T 经典 21）
             threshold ∈ {-0.05,0,0.05}（范围 -0.05–0.05，步长 0.05）

walk-forward 流程（锚定式滚动，样本内逐年加长、样本外各 1 年不重叠）：
  fold1: 样本内 2020-01-01 ~ 2022-12-31 选参 → 样本外 2023 年验证
  fold2: 样本内 2020-01-01 ~ 2023-12-31 选参 → 样本外 2024 年验证
  fold3: 样本内 2020-01-01 ~ 2024-12-31 选参 → 样本外 2025 年验证
  每个 fold 按样本内**总收益率**选优；同时**全部**组合都在样本外跑一遍，
  以计算样本内-样本外秩相关等诊断（"选优的样本外结果"是唯一诚实的评分）。

过拟合诊断指标：
  - Spearman ρ（组合样本内均值 vs 样本外均值）：接近 1 参数稳定，≤0 不可外推；
  - 样本内最优的样本外分位（均值）：掉到中位数以下是典型过拟合；
  - 邻点粗糙度：收益面上相邻参数格点的平均收益落差（年化口径），越大越像噪声拟合；
  - 样本内最优的轴邻点落差：把最优参数沿每根参数轴挪一步后表现差多少，
    "最优参数相邻点剧烈波动"的直接度量；
  - 翻转率：样本内前 1/4 的组合中有多少掉进样本外后 1/4。

输出（output/optimize/<策略>/，图表与报告均标注免责声明）：
  <策略>_<代码>_grid.csv        全网格逐 fold 指标（样本内/样本外）
  <策略>_<代码>_walkforward.csv fold 级选参与样本外结果
  <策略>_<代码>_diagnostics.csv 过拟合诊断与稳健参数选择
  <策略>_<代码>_heatmap.png     样本内 vs 样本外热力图（双面板，第三维取均值）
  <策略>_<代码>_scatter.png     收益-回撤散点图（样本内/样本外两色）
  summary_walkforward.csv       全局汇总（4 策略 × 4 标的）

口径与 scripts/backtest.py 完全一致：qfq 前复权、t 日收盘信号 t+1 日开盘
成交、买入佣金万三/卖出佣金万三+印花税万五、初始 10 万整手（--fractional
切换小数股全仓的学术口径）。

⚠ 全部内容仅为历史模拟，用于学习方法验证；历史模拟回测结果不代表未来表现，
不构成投资建议。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/optimize.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    _setup_cjk_font,
    compute_metrics,
    load_stock,
    run_backtest,
    trade_stats,
)
from strategies import create_strategy  # noqa: E402

DEFAULT_SYMBOLS = ["000001", "600519", "600036", "300750"]

# 每策略的网格：参数名 → 取值列表（范围与步长见模块 docstring）
GRIDS: dict[str, dict] = {
    "ma_cross": {
        "short": [3, 5, 10, 15, 20],
        "long": [20, 30, 60, 120, 250],
    },
    "rsi": {
        "period": [7, 14, 21, 28],
        "oversold": [20.0, 25.0, 30.0, 35.0],
        "exit_level": [50.0, 60.0, 70.0],
    },
    "bollinger": {
        "period": [10, 20, 50],
        "num_std": [1.5, 2.0, 2.5, 3.0],
        "trend_period": [0, 60, 120],
    },
    "momentum": {
        "lookback": [60, 120, 250],
        "skip": [0, 5, 21],
        "threshold": [-0.05, 0.0, 0.05],
    },
}

# 热力图坐标轴：(横轴参数, 纵轴参数)；其余维度在热力图上取均值
HEATMAP_AXES: dict[str, tuple[tuple[str, str], list[str]]] = {
    "ma_cross": (("short", "long"), []),
    "rsi": (("period", "oversold"), ["exit_level"]),
    "bollinger": (("period", "num_std"), ["trend_period"]),
    "momentum": (("lookback", "skip"), ["threshold"]),
}


def build_folds(start: str, end: str, train_years: int, test_years: int) -> list[tuple[str, str, str, str]]:
    """锚定式 walk-forward 折：(样本内起, 样本内止, 样本外起, 样本外止)，YYYYMMDD。

    样本内起点固定为整段起点，每折样本内窗口加长 test_years、样本外窗口向后
    平移 test_years；样本外不足一个完整 test_years 的尾段丢弃。
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    train_span = pd.DateOffset(years=train_years)
    test_span = pd.DateOffset(years=test_years)
    one_day = pd.Timedelta(days=1)
    folds: list[tuple[str, str, str, str]] = []
    anchor = s
    while True:
        train_start = s
        train_end = anchor + train_span - one_day
        test_start = train_end + one_day
        test_end = test_start + test_span - one_day
        if test_end > e:
            break
        folds.append(
            (
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                test_start.strftime("%Y%m%d"),
                test_end.strftime("%Y%m%d"),
            )
        )
        anchor = anchor + test_span
    return folds


def grid_combos(name: str) -> list[tuple[dict, object]]:
    """生成网格内全部合法组合：[(参数字典, 策略实例)]。

    跨参数校验不通过的组合（如 ma_cross 的 short>=long）直接剔除。
    """
    keys = list(GRIDS[name].keys())
    out: list[tuple[dict, object]] = []
    for values in itertools.product(*(GRIDS[name][k] for k in keys)):
        combo = dict(zip(keys, values))
        try:
            strat = create_strategy(name, **combo)
        except ValueError:
            continue  # 非法组合（short>=long 等）
        out.append((combo, strat))
    return out


def fmt_date(ts: str) -> str:
    """YYYYMMDD → YYYY-MM-DD。"""
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def neighbor_roughness(surface: pd.DataFrame) -> float:
    """2D 收益面的邻点粗糙度：所有与上/下/左/右相邻格点的收益落差均值（年化口径）。"""
    vals = surface.to_numpy(dtype=float)
    diffs: list[float] = []
    n_rows, n_cols = vals.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(vals[i, j]):
                continue
            for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_rows and 0 <= nj < n_cols and not np.isnan(vals[ni, nj]):
                    diffs.append(abs(float(vals[i, j] - vals[ni, nj])))
    return float(np.mean(diffs)) if diffs else np.nan


def axis_neighbors(combo: dict, grid: dict) -> list[dict]:
    """把 combo 沿每根参数轴挪 ±1 步的全部相邻组合（非法组合不含在内）。"""
    keys = list(grid.keys())
    out: list[dict] = []
    for k in keys:
        values = grid[k]
        idx = values.index(combo[k])
        for delta in (-1, 1):
            if 0 <= idx + delta < len(values):
                nb = dict(combo)
                nb[k] = values[idx + delta]
                out.append(nb)
    return out


def _neighbor_max_delta(grid_df: pd.DataFrame, combo: dict, column: str) -> float:
    """combo 与其轴邻点在指定指标列上的最大落差；无合法邻点时返回 NaN。"""
    keys = list(GRIDS[grid_df.iloc[0]["strategy"]])
    by_key = {tuple(r[k] for k in keys): i for i, r in grid_df.iterrows()}
    key = tuple(combo[k] for k in keys)
    if key not in by_key:
        return np.nan
    own = float(grid_df.at[by_key[key], column])
    deltas = []
    for nb in axis_neighbors(combo, GRIDS[grid_df.iloc[0]["strategy"]]):
        t = tuple(nb[k] for k in keys)
        if t in by_key:
            deltas.append(abs(float(grid_df.at[by_key[t], column]) - own))
    return float(max(deltas)) if deltas else np.nan


def run_one(name: str, symbol: str, args: argparse.Namespace, folds: list[tuple[str, str, str, str]]) -> dict:
    """对一个（策略, 标的）跑完整 walk-forward：全网格 × 全部 fold 的样本内/样本外回测、
    诊断、CSV 与两张图。返回全局汇总行。"""
    combos = grid_combos(name)
    warmup_max = max(int(strat.warmup) for _, strat in combos)
    df, data_source = load_stock(symbol, args.start, args.end, "qfq", args.datadir,
                                 True, 3, warmup_max)
    print(f"[ok] {name} × {symbol}: 行情 {len(df)} 行，来自 {data_source}；"
          f"网格 {len(combos)} 组合 × {len(folds)} fold")

    rows: list[dict] = []
    for combo, strat in combos:
        row: dict = {"strategy": name, "symbol": symbol, "fractional": args.fractional}
        row.update(combo)
        for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
            eq, trades, eval_s, _ = run_backtest(
                df, strat, tr_s, tr_e, args.capital, args.commission, args.stamp, args.fractional)
            m = compute_metrics(eq, eval_s)
            m.update(trade_stats(trades))
            eqo, tradeso, eval_so, _ = run_backtest(
                df, strat, te_s, te_e, args.capital, args.commission, args.stamp, args.fractional)
            mo = compute_metrics(eqo, eval_so)
            mo.update(trade_stats(tradeso))
            row.update({
                f"fold{fi}_is_total": m["total"], f"fold{fi}_is_annual": m["annual"],
                f"fold{fi}_is_mdd": m["mdd"], f"fold{fi}_is_trades": m["trades"],
                f"fold{fi}_is_win_rate": m["win_rate"],
                f"fold{fi}_oos_total": mo["total"], f"fold{fi}_oos_annual": mo["annual"],
                f"fold{fi}_oos_mdd": mo["mdd"], f"fold{fi}_oos_trades": mo["trades"],
                f"fold{fi}_oos_win_rate": mo["win_rate"],
            })
        rows.append(row)

    g = pd.DataFrame(rows)
    n_folds = len(folds)
    n_combos = len(g)
    # fold 内排名（1=最好）
    for fi in range(1, n_folds + 1):
        g[f"fold{fi}_is_rank"] = g[f"fold{fi}_is_total"].rank(ascending=False, method="min")
        g[f"fold{fi}_oos_rank"] = g[f"fold{fi}_oos_total"].rank(ascending=False, method="min")
    for col in ("total", "annual", "mdd", "trades"):
        g[f"mean_is_{col}"] = g[[f"fold{fi}_is_{col}" for fi in range(1, n_folds + 1)]].mean(axis=1)
        g[f"mean_oos_{col}"] = g[[f"fold{fi}_oos_{col}" for fi in range(1, n_folds + 1)]].mean(axis=1)
    g["mean_is_rank"] = g[[f"fold{fi}_is_rank" for fi in range(1, n_folds + 1)]].mean(axis=1)
    g["mean_oos_rank"] = g[[f"fold{fi}_oos_rank" for fi in range(1, n_folds + 1)]].mean(axis=1)
    g["oos_topq_count"] = sum(
        (g[f"fold{fi}_oos_rank"] <= max(1, n_combos // 4)).astype(int)
        for fi in range(1, n_folds + 1)
    )

    keys = list(GRIDS[name].keys())
    combo_index = {tuple(r[k] for k in keys): i for i, r in g.iterrows()}
    default_params = create_strategy(name).describe()["params"]

    # ---------- 诊断 ----------
    # Spearman ρ = 平均秩上的 Pearson 相关（不依赖 scipy）
    rho = float(g["mean_is_total"].rank(method="average")
                .corr(g["mean_oos_total"].rank(method="average")))
    is_best_idx = int(g["mean_is_total"].idxmax())
    robust_idx = int(g.sort_values(["mean_oos_annual", "mean_is_annual"],
                                   ascending=[False, False]).index[0])
    is_best_combo = {k: g.at[is_best_idx, k] for k in keys}
    robust_combo = {k: g.at[robust_idx, k] for k in keys}
    default_idx = combo_index[tuple(default_params[k] for k in keys)]

    is_topq = g["mean_is_total"] >= g["mean_is_total"].quantile(0.75)
    oos_botq = g["mean_oos_total"] <= g["mean_oos_total"].quantile(0.25)
    flip_rate = float((is_topq & oos_botq).mean())

    axes, avg_dims = HEATMAP_AXES[name]
    surf_is = g.pivot_table(index=axes[1], columns=axes[0], values="mean_is_annual",
                            aggfunc="mean").sort_index(ascending=True)
    surf_oos = g.pivot_table(index=axes[1], columns=axes[0], values="mean_oos_annual",
                             aggfunc="mean").sort_index(ascending=True)
    rough_is = neighbor_roughness(surf_is)
    rough_oos = neighbor_roughness(surf_oos)
    nb_delta_is = _neighbor_max_delta(g, is_best_combo, "mean_is_annual")
    nb_delta_oos = _neighbor_max_delta(g, is_best_combo, "mean_oos_annual")
    nb_count = len([1 for nb in axis_neighbors(is_best_combo, GRIDS[name])
                    if tuple(nb[k] for k in keys) in combo_index])

    # fold 级选参结果
    wf_rows = []
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
    suffix = "_frac" if args.fractional else ""
    outdir = Path(args.outdir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{name}_{symbol}{suffix}"

    grid_cols = (["strategy", "symbol", "fractional"] + keys
                 + [f"fold{fi}_{side}_{m}" for fi in range(1, n_folds + 1)
                    for side in ("is", "oos")
                    for m in ("total", "annual", "mdd", "trades", "win_rate")]
                 + [f"fold{fi}_{side}_rank" for fi in range(1, n_folds + 1) for side in ("is", "oos")]
                 + ["mean_is_total", "mean_is_annual", "mean_is_mdd", "mean_is_trades",
                    "mean_oos_total", "mean_oos_annual", "mean_oos_mdd", "mean_oos_trades",
                    "mean_is_rank", "mean_oos_rank", "oos_topq_count"])
    g[grid_cols].to_csv(outdir / f"{base}_grid.csv", index=False, encoding="utf-8-sig")
    wf.to_csv(outdir / f"{base}_walkforward.csv", index=False, encoding="utf-8-sig")

    robust_row = g.loc[robust_idx]
    default_row = g.loc[default_idx]
    diag = pd.DataFrame([
        ("n_combos", n_combos), ("n_folds", n_folds), ("fractional", args.fractional),
        ("data_source", data_source),
        ("spearman_rho_is_oos", rho),
        ("is_best_params", ",".join(f"{k}={is_best_combo[k]}" for k in keys)),
        ("is_best_mean_is_annual", g.at[is_best_idx, "mean_is_annual"]),
        ("is_best_mean_oos_total", g.at[is_best_idx, "mean_oos_total"]),
        ("is_best_mean_oos_annual", g.at[is_best_idx, "mean_oos_annual"]),
        ("is_best_mean_oos_pct", is_best_mean_oos_pct),
        ("is_best_neighbor_count", nb_count),
        ("is_best_neighbor_max_delta_is_annual", nb_delta_is),
        ("is_best_neighbor_max_delta_oos_annual", nb_delta_oos),
        ("robust_params", ",".join(f"{k}={robust_combo[k]}" for k in keys)),
        ("robust_mean_is_annual", robust_row["mean_is_annual"]),
        ("robust_mean_oos_annual", robust_row["mean_oos_annual"]),
        ("robust_mean_oos_rank", robust_row["mean_oos_rank"]),
        ("robust_oos_topq_count", int(robust_row["oos_topq_count"])),
        ("default_params", ",".join(f"{k}={default_params[k]}" for k in keys)),
        ("default_mean_is_annual", default_row["mean_is_annual"]),
        ("default_mean_oos_annual", default_row["mean_oos_annual"]),
        ("robust_vs_default_oos_delta", robust_row["mean_oos_annual"] - default_row["mean_oos_annual"]),
        ("flip_rate_is_topq_to_oos_botq", flip_rate),
        ("roughness_is_annual", rough_is),
        ("roughness_oos_annual", rough_oos),
        ("disclaimer", DISCLAIMER),
    ], columns=["metric", "value"])
    diag.to_csv(outdir / f"{base}_diagnostics.csv", index=False, encoding="utf-8-sig")

    if not args.no_plot:
        _plot_heatmap(g, name, symbol, axes, avg_dims, is_best_combo, robust_combo,
                      outdir / f"{base}_heatmap.png", args.fractional)
        _plot_scatter(g, name, symbol, is_best_combo, robust_combo, default_params,
                      rho, is_best_mean_oos_pct, flip_rate, outdir / f"{base}_scatter.png",
                      args.fractional)

    # ---------- 控制台小结 ----------
    print(f"\n===== walk-forward: {name} × {symbol}（{len(folds)} fold，{n_combos} 组合）=====")
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
    print(f"稳健参数（样本外年化均值最优） {','.join(f'{k}={robust_combo[k]}' for k in keys)}："
          f"样本外年化均值 {_pct(robust_row['mean_oos_annual'])}（默认参数 "
          f"{_pct(default_row['mean_oos_annual'])}），样本外平均排名 {robust_row['mean_oos_rank']:.1f}/{n_combos}")
    print(f"邻点粗糙度：样本内 {_pct(rough_is)}/步，样本外 {_pct(rough_oos)}/步；"
          f"样本内最优的轴邻点最大落差：样本内 {_pct(nb_delta_is)}，样本外 {_pct(nb_delta_oos)}；"
          f"翻转率 {flip_rate:.0%}")
    print(f"⚠ {DISCLAIMER}")

    return {
        "strategy": name, "symbol": symbol, "fractional": args.fractional,
        "n_combos": n_combos, "n_folds": n_folds,
        "folds": "; ".join(f"{fmt_date(tr_s)}~{fmt_date(tr_e)} → {fmt_date(te_s)}~{fmt_date(te_e)}"
                          for tr_s, tr_e, te_s, te_e in folds),
        "is_best_per_fold": "; ".join(r["is_best_params"] for r in wf_rows),
        "is_best_oos_per_fold": "; ".join(f"{r['is_best_oos_total']:.4f}" for r in wf_rows),
        "is_best_mean_oos_total": is_best_mean_oos,
        "is_best_mean_oos_pct": is_best_mean_oos_pct,
        "spearman_rho": rho,
        "robust_params": ",".join(f"{k}={robust_combo[k]}" for k in keys),
        "robust_mean_is_annual": robust_row["mean_is_annual"],
        "robust_mean_oos_annual": robust_row["mean_oos_annual"],
        "robust_mean_oos_rank": robust_row["mean_oos_rank"],
        "robust_oos_topq_count": int(robust_row["oos_topq_count"]),
        "default_params": ",".join(f"{k}={default_params[k]}" for k in keys),
        "default_mean_is_annual": default_row["mean_is_annual"],
        "default_mean_oos_annual": default_row["mean_oos_annual"],
        "robust_vs_default_oos_delta": robust_row["mean_oos_annual"] - default_row["mean_oos_annual"],
        "flip_rate": flip_rate,
        "roughness_is_annual": rough_is,
        "roughness_oos_annual": rough_oos,
        "is_best_neighbor_max_delta_is": nb_delta_is,
        "is_best_neighbor_max_delta_oos": nb_delta_oos,
    }


# ---------------------------------------------------------------------------
# 绘图：热力图 + 收益-回撤散点图
# ---------------------------------------------------------------------------

def _plot_heatmap(g: pd.DataFrame, name: str, symbol: str, axes: tuple[str, str],
                  avg_dims: list[str], is_best_combo: dict, robust_combo: dict,
                  out_png: Path, fractional: bool) -> None:
    """样本内 vs 样本外双面板热力图（第三维参数取均值，年化收益率 ×100）。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    x_param, y_param = axes
    surf_is = g.pivot_table(index=y_param, columns=x_param, values="mean_is_annual",
                            aggfunc="mean").sort_index(ascending=True)
    surf_oos = g.pivot_table(index=y_param, columns=x_param, values="mean_oos_annual",
                             aggfunc="mean").sort_index(ascending=True)
    agg_note = f"，其余维度取均值({'+'.join(avg_dims)})" if avg_dims else ""

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    x_labels = [str(v) for v in surf_is.columns]
    y_labels = [str(v) for v in surf_is.index]
    panels = (("样本内（各 fold 年化收益均值 %）", surf_is), ("样本外（各 fold 年化收益均值 %）", surf_oos))
    all_vals = np.concatenate([surf_is.to_numpy().ravel(), surf_oos.to_numpy().ravel()])
    all_vals = all_vals[~np.isnan(all_vals)] * 100
    if len(all_vals) == 0:
        print(f"[warn] {symbol} {name}: 热力图无有效数据，跳过", file=sys.stderr)
        return
    vmax = max(0.0, float(np.nanmax(all_vals)))
    vmin = min(0.0, float(np.nanmin(all_vals)))
    # TwoSlopeNorm 要求 vmin < vcenter < vmax 严格成立；全正/全负/全零面时收缩区间
    if vmin >= 0.0:
        vmin = -max(vmax * 0.1, 0.01)
    if vmax <= 0.0:
        vmax = max(-vmin * 0.1, 0.01)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

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
                    ax.text(j, i, f"{masked[i, j]:.0f}", ha="center", va="center",
                            fontsize=8, color="black")
        if (is_best_combo[x_param] in surf.columns
                and is_best_combo[y_param] in surf.index):
            bx = x_labels.index(str(is_best_combo[x_param]))
            by = y_labels.index(str(is_best_combo[y_param]))
            ax.add_patch(Rectangle((bx - 0.5, by - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=2.5))
        if (robust_combo[x_param] in surf.columns
                and robust_combo[y_param] in surf.index):
            rx = x_labels.index(str(robust_combo[x_param]))
            ry = y_labels.index(str(robust_combo[y_param]))
            ax.plot(rx, ry, marker="D", markersize=9, mfc="none", mec="blue", mew=2)

    cax = fig.add_axes([0.93, 0.14, 0.012, 0.70])
    fig.colorbar(im, cax=cax, label="年化收益率（%）")
    kind = "（小数股全仓口径）" if fractional else "（10 万整手口径）"
    fig.suptitle(f"{symbol} · {name}{agg_note}：样本内 vs 样本外参数热力图{kind}\n"
                 f"■=样本内全局最优  ◇=稳健参数（样本外均值最优）")
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 热力图已保存: {out_png}")


def _plot_scatter(g: pd.DataFrame, name: str, symbol: str, is_best_combo: dict,
                  robust_combo: dict, default_params: dict, rho: float,
                  is_best_mean_oos_pct: float, flip_rate: float,
                  out_png: Path, fractional: bool) -> None:
    """收益-回撤散点图：每组合一个样本内点 + 一个样本外点（年化收益 vs 最大回撤）。"""
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.scatter(g["mean_is_annual"] * 100, g["mean_is_mdd"] * 100,
               s=26, alpha=0.75, color="#1f77b4", label="样本内（3 fold 均值）")
    ax.scatter(g["mean_oos_annual"] * 100, g["mean_oos_mdd"] * 100,
               s=26, alpha=0.75, color="#ff7f0e", label="样本外（3 fold 均值）")

    keys = list(GRIDS[name].keys())
    combo_index = {tuple(r[k] for k in keys): i for i, r in g.iterrows()}
    is_best_i = combo_index[tuple(is_best_combo[k] for k in keys)]
    robust_i = combo_index[tuple(robust_combo[k] for k in keys)]
    default_i = combo_index[tuple(default_params[k] for k in keys)]

    ax.scatter([g.at[is_best_i, "mean_is_annual"] * 100], [g.at[is_best_i, "mean_is_mdd"] * 100],
               marker="s", s=150, facecolors="none", edgecolors="black", lw=2,
               label="样本内全局最优（■）")
    ax.scatter([g.at[robust_i, "mean_oos_annual"] * 100], [g.at[robust_i, "mean_oos_mdd"] * 100],
               marker="D", s=130, facecolors="none", edgecolors="darkblue", lw=2,
               label="稳健参数（◇）")
    ax.scatter([g.at[default_i, "mean_oos_annual"] * 100], [g.at[default_i, "mean_oos_mdd"] * 100],
               marker="^", s=120, facecolors="none", edgecolors="darkgreen", lw=2,
               label="默认参数（△）")
    ax.annotate("", xy=(g.at[is_best_i, "mean_oos_annual"] * 100, g.at[is_best_i, "mean_oos_mdd"] * 100),
                xytext=(g.at[is_best_i, "mean_is_annual"] * 100, g.at[is_best_i, "mean_is_mdd"] * 100),
                arrowprops=dict(arrowstyle="->", color="black", lw=1, linestyle=":"))
    ax.text(0.03, 0.03,
            f"Spearman ρ = {rho:+.2f}\n样本内最优的平均样本外分位 = {is_best_mean_oos_pct:.0%}\n"
            f"翻转率（样本内前1/4→样本外后1/4） = {flip_rate:.0%}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("年化收益率（%）")
    ax.set_ylabel("最大回撤（%）")
    kind = "（小数股全仓口径）" if fractional else "（10 万整手口径）"
    ax.set_title(f"{symbol} · {name} 收益-回撤散点图{kind}\n虚线箭头 = 样本内全局最优 → 其在样本外的表现")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[ok] 收益-回撤散点图已保存: {out_png}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="参数网格扫描 + walk-forward 样本外验证（4 策略 × 4 标的），"
                    "输出逐 fold 选参结果、过拟合诊断、热力图与收益-回撤散点图",
        epilog=(
            "示例:\n"
            "  python scripts/optimize.py\n"
            "  python scripts/optimize.py --strategy ma_cross --symbols 000001\n"
            "  python scripts/optimize.py --fractional --symbols 600519\n"
            "所有输出仅为历史模拟，不构成投资建议。"
        ),
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="股票代码，默认 000001 600519 600036 300750")
    parser.add_argument("--strategy", nargs="+", default=sorted(GRIDS),
                        help="策略名，默认全部 4 个")
    parser.add_argument("--start", default="20200101", help="整段回测起点 YYYYMMDD，默认 20200101")
    parser.add_argument("--end", default="20251231", help="整段回测终点 YYYYMMDD，默认 20251231")
    parser.add_argument("--train-years", type=int, default=3,
                        help="首个样本内窗口年数，默认 3（之后每折加长 1 个样本外窗口）")
    parser.add_argument("--test-years", type=int, default=1, help="每个样本外窗口年数，默认 1")
    parser.add_argument("--datadir", default="data/daily", help="行情 CSV 目录，默认 data/daily")
    parser.add_argument("--outdir", default="output/optimize", help="输出目录，默认 output/optimize")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金，默认 100000")
    parser.add_argument("--commission", type=float, default=0.0003, help="单边佣金，默认万三")
    parser.add_argument("--stamp", type=float, default=0.0005, help="卖出印花税，默认万五")
    parser.add_argument("--fractional", action="store_true",
                        help="小数股全仓（学术口径，研究信号本身；默认 A 股整手）")
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 图")
    args = parser.parse_args(argv)
    unknown = set(args.strategy) - set(GRIDS)
    if unknown:
        parser.error(f"未知策略: {', '.join(sorted(unknown))}；可选: {', '.join(sorted(GRIDS))}")
    if args.train_years < 1 or args.test_years < 1:
        parser.error("--train-years/--test-years 必须为正整数")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    args = parse_args(argv)
    folds = build_folds(args.start, args.end, args.train_years, args.test_years)
    if not folds:
        print(f"[error] {args.start}~{args.end} 放不下至少一个 "
              f"{args.train_years}+{args.test_years} 年的 walk-forward fold", file=sys.stderr)
        return 2

    print(f"walk-forward 区间: {args.start} ~ {args.end} | 策略: {', '.join(args.strategy)} | "
          f"标的: {' '.join(args.symbols)} | 初始资金 {args.capital:,.0f} | "
          f"佣金 {args.commission:.2%} + 印花税 {args.stamp:.2%} | "
          f"口径: {'小数股全仓' if args.fractional else '整手(1手=100股)'}")
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        print(f"  fold{fi}: 样本内 {fmt_date(tr_s)}~{fmt_date(tr_e)} 选参 → "
              f"样本外 {fmt_date(te_s)}~{fmt_date(te_e)} 验证")
    print(f"⚠ {DISCLAIMER}\n")

    summary_rows: list[dict] = []
    for name in args.strategy:
        for symbol in args.symbols:
            try:
                summary_rows.append(run_one(name, symbol, args, folds))
            except Exception as exc:  # noqa: BLE001 单格失败不中断批量任务
                print(f"[failed] {name} × {symbol}: {exc}", file=sys.stderr)

    if summary_rows:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        summary["disclaimer"] = DISCLAIMER
        summary_path = outdir / "summary_walkforward.csv"
        # 合并已有汇总（如整手口径与小数股口径分两次运行），按 (策略, 标的, 口径) 去重
        if summary_path.exists():
            try:
                old = pd.read_csv(summary_path)
                key_cols = ["strategy", "symbol", "fractional"]
                # 归一化键列类型：CSV 回读时 000001 会变成整数 1，必须先还原再比较
                for frame in (old, summary):
                    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
                    frame["fractional"] = frame["fractional"].astype(bool)
                old = old[~old.set_index(key_cols).index.isin(summary.set_index(key_cols).index)]
                summary = pd.concat([old, summary], ignore_index=True)
            except Exception as exc:  # noqa: BLE001 旧汇总损坏时直接重建
                print(f"[warn] 旧汇总读取失败，直接重建: {exc}", file=sys.stderr)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[ok] 全局汇总已保存: {summary_path}")

    print(f"\n⚠ {DISCLAIMER}")
    return 0 if len(summary_rows) == len(args.strategy) * len(args.symbols) else 1


if __name__ == "__main__":
    sys.exit(main())
