"""259 个开源因子的 A 股有效性体检：横截面 IC 筛选（issue #43）。

任务来源：GitHub issue #43（issue #36 因子移植的后续）。对已移植的
WorldQuant Alpha101（101 个）+ Qlib Alpha158 量价部分（158 个）逐个计算
横截面 IC（Information Coefficient）并汇总排序，作为有效性体检：

- **IC 定义**：每个交易日 t，用截至 t 的信息计算因子值（alpha_factors.py
  的窗口全部右端对齐当日，无未来函数），与该股 t 日收盘买入、个股自身
  第 t+h 个交易日收盘卖出的未来收益（h=1、5 两个口径）做**横截面
  Spearman 秩相关**，得到当日 IC_t；全样本汇总 IC 均值、IC 标准差、
  ICIR（IC 均值/IC 标准差）、年化 ICIR（×√(250/h)）、t 值、IC>0 占比、
  |IC|>2% 占比等。
- **数据**：本机 TSP parquet（kline_daily，2019-07-19 起），qfq 价由
  adj_factor 换算（screener._qfq_factors，与 screener.py 同口径）。
  股票池取窗口内出现过的全部股票（含已退市，缓解幸存者偏差），按历史
  交易日数过滤后按代码等距抽样（--max-stocks）。
- **行业/市值依赖的 19 个 alpha**（18 个 IndNeutralize + alpha056）：
  用本地静态快照近似补齐——同花顺行业分类（ext_data/ext_hy_ths）、
  总股本（instruments.parquet，2026-11-17 快照）。行业为快照口径
  （忽略历史行业变更）、cap 用快照总股本（忽略历史股本变动），
  局限性如实记录（见优化日志第 19 章）。
- **无未来函数**：因子值只含当日及之前信息；未来收益严格取 t 之后的
  交易日（停牌跳过，按个股自身交易日），最后 h 个交易日无未来收益
  → IC 为 NaN 不参与汇总。

用法：
    python scripts/factor_ic.py selftest    # 合成数据自检（不访问网络）
    # 全流程：从 TSP parquet 导入 → 抽样 → 计算 259 因子 → IC 汇总
    python scripts/factor_ic.py run --max-stocks 1200 --start 20190719 --end 20261231
    # 复用已导入的 data/daily CSV（--no-fetch）
    python scripts/factor_ic.py run --no-fetch --max-stocks 1200

输出（output/factors/）：
    ic_alpha101_h1_summary.csv   Alpha101 全 101 个 × 1 日未来收益（按 |年化 ICIR| 排序）
    ic_alpha101_h5_summary.csv   同上，5 日未来收益口径
    ic_alpha158_h1_summary.csv   Alpha158 全 158 个 × 1 日未来收益
    ic_alpha158_h5_summary.csv   同上，5 日口径
    ic_series_top20_h1.csv       全体 259 因子中 |IC 均值| Top-20 的逐日 IC 序列（1d）
    ic_alpha101_ranking_h1.png   Alpha101 Top-10/Bottom-10 IC 均值条形图
    ic_alpha158_ranking_h1.png   Alpha158 Top-10/Bottom-10 IC 均值条形图

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/factor_ic.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from alpha_factors import (  # noqa: E402
    ALPHA101_NAMES,
    ALPHA158_NAMES,
    compute_alpha101,
    compute_alpha158,
    load_frames,
    _map_cols,
)
from backtest import DISCLAIMER, standardize_stock  # noqa: E402
import screener  # noqa: E402

# ---------------------------------------------------------------------------
# 常量与默认值
# ---------------------------------------------------------------------------

TSP_DIR_DEFAULT = screener.TSP_DIR_DEFAULT
DATADIR_DEFAULT = "data/daily"
OUTDIR_DEFAULT = "output/factors"
MIN_ROWS_DEFAULT = 250      # 单股最少交易日（约 1 年，用于股票池过滤）
MIN_PAIRS_DEFAULT = 30      # 单日横截面最少有效配对（因子值+未来收益都非 NaN）
MIN_IC_DATES_DEFAULT = 60   # 因子汇总最少有效 IC 日期数
CHUNK_101 = 12              # Alpha101 分块大小（内存/耗时折中）
CHUNK_158 = 16              # Alpha158 分块大小

# 汇总输出列（按 |年化 ICIR| 排序，写 CSV 时按此顺序落盘）
SUMMARY_COLS = ["factor", "lib", "horizon", "n_dates", "ic_mean", "ic_std",
                "icir", "icir_annualized", "t_stat", "ic_gt0_ratio",
                "abs_ic_gt_2pct", "ic_median", "ic_min", "ic_max"]


# ---------------------------------------------------------------------------
# IC 计算
# ---------------------------------------------------------------------------

def _fwd_shift(a: np.ndarray, h: int) -> np.ndarray:
    """个股自身交易日的未来第 h 个交易日取值（尾部补 NaN）。"""
    n = len(a)
    if h >= n:
        return np.full(n, np.nan)
    return np.concatenate([a[h:], np.full(h, np.nan)])


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """未来 h 个交易日收益（t 日收盘买入、个股自身第 t+h 个交易日收盘卖出）。

    逐列压缩掉 NaN（停牌日）后在个股自身连续交易日上平移，与因子库的
    时间序列算子同口径：停牌日的未来收益按下一个实际交易日计算。
    最后 h 个交易日无未来行情 → NaN（不参与 IC）。
    """
    def fwd(a: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return _fwd_shift(a, h) / a - 1.0
    return _map_cols(close, fwd)


def _rank_cs(frame: pd.DataFrame) -> pd.DataFrame:
    """横截面秩（axis=1，NaN 保持 NaN），与 alpha_factors._rank_cs 同口径。"""
    return frame.rank(axis=1)


def _rank_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.rank(axis=1).to_numpy(dtype=float)


def cross_sectional_ic(factor: pd.DataFrame, ret_rank: np.ndarray,
                       min_pairs: int = MIN_PAIRS_DEFAULT) -> pd.Series:
    """每个交易日的横截面 Spearman IC（秩相关）。

    factor 为 日期 × 股票 宽表；ret_rank 为未来收益的横截面秩矩阵
    （ndarray，同一形状，预先算好一次）。Spearman 即两列秩的 Pearson
    相关：**去均值与求和都在两侧共同有效的配对（交集）上进行**
    （pandas corr 同口径——两侧 NaN 模式不同的股票/日期不做单侧近似）。
    当日有效配对 < min_pairs 时 IC 置 NaN（横截面太薄不采信）。
    """
    f_rank = _rank_matrix(factor)
    r = ret_rank
    if f_rank.shape != r.shape:
        raise ValueError(
            f"因子矩阵形状 {f_rank.shape} 与未来收益秩矩阵 {r.shape} 不一致"
            f"（检查 cap/industry 等对齐序列是否 reindex 到面板列）")
    mask = ~np.isnan(f_rank) & ~np.isnan(r)
    n = mask.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_mean = np.where(mask, f_rank, 0.0).sum(axis=1) / n
        r_mean = np.where(mask, r, 0.0).sum(axis=1) / n
        f_c = np.where(mask, f_rank - f_mean[:, None], 0.0)
        r_c = np.where(mask, r - r_mean[:, None], 0.0)
        num = (f_c * r_c).sum(axis=1)
        den = np.sqrt((f_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1))
        ic = num / den
    ic = np.where(n >= min_pairs, ic, np.nan)
    return pd.Series(ic, index=factor.index)


def summarize_ic(ic: pd.Series, horizon: int,
                 min_dates: int = MIN_IC_DATES_DEFAULT) -> dict | None:
    """单因子 IC 序列 → 汇总统计。有效日期不足时返回 None（不采信）。"""
    ic = ic.dropna()
    n = int(len(ic))
    if n < min_dates:
        return None
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    if std > 0:
        ir = mean / std
        ir_ann = ir * math.sqrt(250.0 / horizon)
        t_stat = ir * math.sqrt(n)
    else:
        ir = ir_ann = t_stat = np.nan
    return {
        "n_dates": n, "ic_mean": mean, "ic_std": std, "icir": ir,
        "icir_annualized": ir_ann, "t_stat": t_stat,
        "ic_gt0_ratio": float((ic > 0).mean()),
        "abs_ic_gt_2pct": float((ic.abs() > 0.02).mean()),
        "ic_median": float(ic.median()),
        "ic_min": float(ic.min()), "ic_max": float(ic.max()),
    }


# ---------------------------------------------------------------------------
# 数据导入（TSP parquet → data/daily/<代码>_qfq.csv，与 screener.py 同口径）
# ---------------------------------------------------------------------------

def _load_industry(tsp_dir: str) -> pd.Series:
    """同花顺行业分类静态快照 → Series（code → 行业标签）。

    文件 ext_data/ext_hy_ths/part.parquet 最后一列为行业标签；
    无该文件时返回空 Series（19 个行业/市值 alpha 将全 NaN，如实告警）。
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        raw = pq.read_table(Path(tsp_dir) / "ext_data" / "ext_hy_ths"
                            / "part.parquet").to_pandas()
    except Exception as exc:  # noqa: BLE001 缺失时降级，不中断主流程
        print(f"[warn] 同花顺行业快照不可用（{exc}），18 个行业 alpha "
              f"将全为 NaN", file=sys.stderr)
        return pd.Series(dtype=str)
    code = raw["code"].astype(str).str.zfill(6)
    industry = pd.Series(raw.iloc[:, -1].astype(str).values, index=code)
    return industry[~industry.index.duplicated(keep="last")]


def _load_instruments(tsp_dir: str) -> tuple[pd.Series, pd.Series]:
    """instruments 快照 → (float_shares, total_shares) 两个 code → float Series。"""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        raw = pq.read_table(Path(tsp_dir) / "instruments"
                            / "instruments.parquet").to_pandas()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] instruments 快照不可用（{exc}），换手率/cap 留 NaN",
              file=sys.stderr)
        return pd.Series(dtype=float), pd.Series(dtype=float)
    code = raw["code"].astype(str).str.zfill(6)
    out = {}
    for key in ("float_shares", "total_shares"):
        out[key] = pd.to_numeric(raw[key], errors="coerce")
        out[key].index = code
        out[key] = out[key][~out[key].index.duplicated(keep="last")]
    return out["float_shares"], out["total_shares"]


def _import_tsp(tsp_dir: str, start: str, end: str, datadir: str,
                max_stocks: int | None, min_rows: int) -> list[str]:
    """读 TSP kline_daily 面板 → 抽样 → 逐票写 qfq CSV。返回抽样代码列表。

    抽样口径：窗口内交易日数 ≥ min_rows 的全部代码按代码排序后等距抽样
    （跨板块/上市时间均匀铺开）。CSV 已存在且行数足够时跳过（复用缓存）。
    """
    t0 = time.time()
    panel, events = screener._load_tsp_panel(tsp_dir, start, end)
    if panel.empty:
        raise RuntimeError(f"TSP kline_daily 在 {start}~{end} 内无完整分区")
    print(f"[ok] TSP 面板: {len(panel)} 行 × {panel['code'].nunique()} 只，"
          f"覆盖 {panel['date'].min():%Y-%m-%d} ~ {panel['date'].max():%Y-%m-%d}")

    counts = panel.groupby("code").size()
    eligible = counts[counts >= min_rows].sort_index()
    n_eligible = len(eligible)
    if max_stocks and n_eligible > max_stocks:
        stride = math.ceil(n_eligible / max_stocks)
        eligible = eligible.iloc[::stride]
    symbols = list(eligible.index)
    print(f"[info] 股票池: 窗口内 {n_eligible} 只满足 ≥{min_rows} 交易日，"
          f"等距抽样 {len(symbols)} 只")

    float_shares, total_shares = _load_instruments(tsp_dir)
    by_code = {c: g.drop(columns="symbol").reset_index(drop=True)
               for c, g in panel.groupby("code", sort=False)}
    ev_by_code: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not events.empty:
        events["code"] = events["symbol"].astype(str).str.split(".").str[0]
        for c, g in events.groupby("code", sort=False):
            ev_by_code[c] = (g["trade_date"].values, g["ex_factor"].values)

    outdir = Path(datadir)
    outdir.mkdir(parents=True, exist_ok=True)
    new = skipped = 0
    for i, symbol in enumerate(symbols):
        csv_path = outdir / f"{symbol}_qfq.csv"
        sub = by_code.get(symbol)
        if sub is None or sub.empty:
            continue
        if csv_path.exists():
            try:
                if len(standardize_stock(pd.read_csv(csv_path))) >= len(sub):
                    skipped += 1
                    continue
            except Exception:  # noqa: BLE001 损坏缓存重导
                pass
        dates = sub["date"].values
        if symbol in ev_by_code:
            ev_d, ev_f = ev_by_code[symbol]
        else:
            ev_d = np.array([], dtype="datetime64[ns]")
            ev_f = np.array([], dtype=float)
        fac = screener._qfq_factors(dates, ev_d, ev_f)
        close = (sub["close"].values * fac).astype(float)
        open_ = (sub["open"].values * fac).astype(float)
        high = (sub["high"].values * fac).astype(float)
        low = (sub["low"].values * fac).astype(float)
        volume = sub["volume"].values.astype(float)
        prev_close = np.concatenate([[np.nan], close[:-1]])
        if symbol in float_shares:
            with np.errstate(divide="ignore", invalid="ignore"):
                turnover = volume * 10000.0 / float_shares[symbol]
            turnover = np.where(np.isfinite(turnover), turnover, np.nan)
        else:
            turnover = np.nan
        pd.DataFrame({
            "日期": sub["date"].dt.strftime("%Y-%m-%d"),
            "股票代码": symbol,
            "开盘": open_, "收盘": close, "最高": high, "最低": low,
            "成交量": volume, "成交额": sub["amount"].values,
            "振幅": (high - low) / prev_close * 100.0,
            "涨跌幅": (close / prev_close - 1.0) * 100.0,
            "涨跌额": close - prev_close,
            "换手率": turnover,
        })[screener.OUT_COLS].to_csv(csv_path, index=False,
                                     encoding="utf-8-sig")
        new += 1
        if (i + 1) % 200 == 0 or i == len(symbols) - 1:
            print(f"[进度] 导入 {i + 1}/{len(symbols)}：新写 {new}、"
                  f"复用 {skipped}，用时 {time.time() - t0:.0f}s", flush=True)
    print(f"[ok] 导入完成：{len(symbols)} 只（新写 {new}、复用 {skipped}），"
          f"用时 {time.time() - t0:.0f}s")
    return symbols


def _symbols_from_datadir(datadir: str, max_stocks: int | None) -> list[str]:
    """--no-fetch：复用 datadir 下已有 CSV，返回代码列表（排序）。"""
    import re
    symbols = sorted({p.stem.split("_")[0]
                      for p in Path(datadir).glob("*_qfq.csv")
                      if re.fullmatch(r"\d{6}", p.stem.split("_")[0])})
    if not symbols:
        raise RuntimeError(f"{datadir} 下没有任何 *_qfq.csv，无法 --no-fetch")
    if max_stocks and len(symbols) > max_stocks:
        stride = math.ceil(len(symbols) / max_stocks)
        symbols = symbols[::stride]
    print(f"[info] --no-fetch 复用 {len(symbols)} 只已有 CSV")
    return symbols


# ---------------------------------------------------------------------------
# 主流程：逐因子计算 → IC → 汇总
# ---------------------------------------------------------------------------

def _run_library(compute_fn, names: list[str], frames: dict[str, pd.DataFrame],
                 kwargs: dict, chunk: int, label: str, horizons: list[int],
                 ret_ranks: dict[int, np.ndarray], min_pairs: int,
                 min_dates: int) -> tuple[list[dict],
                                          dict[str, dict[int, pd.Series]]]:
    """对一个因子库逐块计算因子值并汇总 IC。

    返回 (summary_rows, ic_series)：ic_series[factor][horizon] = 逐日 IC。
    """
    t0 = time.time()
    rows: list[dict] = []
    series: dict[str, dict[int, pd.Series]] = {}
    for i in range(0, len(names), chunk):
        batch = names[i:i + chunk]
        res = compute_fn(frames, include=batch, **kwargs)
        for name in batch:
            frame = res[name]
            series[name] = {}
            for h in horizons:
                ic = cross_sectional_ic(frame, ret_ranks[h], min_pairs)
                series[name][h] = ic
                summ = summarize_ic(ic, h, min_dates)
                if summ is not None:
                    rows.append({"factor": name, "lib": label,
                                 "horizon": h, **summ})
            del frame
        del res
        el = time.time() - t0
        print(f"[进度] {label}: {min(i + chunk, len(names))}/{len(names)}，"
              f"用时 {el:.0f}s", flush=True)
    print(f"[ok] {label}: {len(names)} 个因子完成，"
          f"{len(rows)} 个通过最少日期门槛，用时 {time.time() - t0:.0f}s")
    return rows, series


def _save_summaries(rows: list[dict], libs: list[str], horizons: list[int],
                    outdir: Path) -> None:
    for lib in libs:
        for h in horizons:
            sub = [r for r in rows if r["lib"] == lib and r["horizon"] == h]
            df = pd.DataFrame(sub).sort_values(
                "icir_annualized", key=lambda s: s.abs(), ascending=False)
            df = df.reindex(columns=SUMMARY_COLS)
            df.to_csv(outdir / f"ic_{lib}_h{h}_summary.csv", index=False,
                      encoding="utf-8-sig")
            if not df.empty:
                print(f"[ok] ic_{lib}_h{h}_summary.csv: {len(df)} 个因子；"
                      f"最强 |年化ICIR| = {df['icir_annualized'].abs().max():.2f}"
                      f"（{df['factor'].iloc[0]}）")


def _top_bottom(rows: list[dict], lib: str, h: int, k: int = 10
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame([r for r in rows if r["lib"] == lib and r["horizon"] == h])
    df = df.sort_values("ic_mean", ascending=False)
    return df.head(k), df.tail(k).sort_values("ic_mean")


def _plot_ranking(rows: list[dict], lib: str, h: int, outdir: Path,
                  k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top, bottom = _top_bottom(rows, lib, h, k)
    both = pd.concat([top, bottom])
    if both.empty:
        return
    labels = both["factor"].tolist()
    means = both["ic_mean"].to_numpy()
    errs = (both["ic_std"] / np.sqrt(both["n_dates"])).to_numpy()
    y = np.arange(len(labels))
    colors = ["#c0392b" if m >= 0 else "#2980b9" for m in means]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(y, means, xerr=errs, color=colors, alpha=0.85, error_kw={"lw": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel(f"IC mean (h={h}d), error bar = std/sqrt(n)")
    ax.set_title(f"{lib} Top-{k}/Bottom-{k} by IC mean")
    ax.invert_yaxis()
    fig.tight_layout()
    path = outdir / f"ic_{lib}_ranking_h{h}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[ok] {path.name}")


def _save_top_series(series: dict[str, dict[int, pd.Series]],
                     rows: list[dict], h: int, outdir: Path, k: int = 20) -> None:
    """全体因子中 |IC 均值| Top-k 的逐日 IC 序列（宽表：日期 × 因子）。"""
    df = pd.DataFrame([r for r in rows if r["horizon"] == h])
    df = df.reindex(df["ic_mean"].abs().sort_values(ascending=False).index)
    top_names = df["factor"].head(k).tolist()
    cols = {name: series[name][h].rename(name) for name in top_names}
    out = pd.concat(cols, axis=1).sort_index()
    out.index.name = "date"
    out.to_csv(outdir / f"ic_series_top{k}_h{h}.csv", encoding="utf-8-sig")
    print(f"[ok] ic_series_top{k}_h{h}.csv: {out.shape[0]} 日 × {out.shape[1]} 因子")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_fetch:
        symbols = _import_tsp(args.tsp_dir, args.start, args.end, args.datadir,
                              args.max_stocks, args.min_rows)
    else:
        symbols = _symbols_from_datadir(args.datadir, args.max_stocks)

    print(f"[info] 加载 {len(symbols)} 只股票日线（{args.datadir}）...")
    frames, _names = load_frames(symbols, datadir=args.datadir, adjust="qfq",
                                 start=args.start, end=args.end)
    close = frames["close"]
    print(f"[ok] 面板: {close.shape[0]} 个交易日 × {close.shape[1]} 只股票")

    # 未来收益及其横截面秩（每个口径只算一次）
    horizons = args.horizons
    ret_ranks: dict[int, np.ndarray] = {}
    for h in horizons:
        ret = forward_returns(close, h)
        ret_ranks[h] = _rank_matrix(ret)
        valid = ret.notna().sum().sum()
        print(f"[info] 未来 {h} 日收益：全样本有效配对 {valid} 个")

    # 行业/市值静态快照（19 个依赖 alpha 的近似输入）。必须 reindex 到
    # 面板列：否则 pandas 对齐会把结果扩到全部快照股票的列（如 alpha056
    # 的 returns × cap），与未来收益矩阵形状不一致。
    cols = close.columns
    industry = _load_industry(args.tsp_dir)
    _float_shares, total_shares = _load_instruments(args.tsp_dir)
    cap = total_shares.reindex(cols) if len(total_shares) else None
    if len(industry) == 0:
        industry = None
        print("[warn] 无行业分类输入：18 个行业 alpha 将全 NaN，不参与排名",
              file=sys.stderr)
    else:
        industry = industry.reindex(cols)
        print(f"[info] 行业分类快照覆盖 {industry.notna().sum()}/{len(cols)} 只"
              f"（面板内股票）")

    rows: list[dict] = []
    series: dict[str, dict[int, pd.Series]] = {}
    libs = []
    if args.only in ("101", "both"):
        r101, s101 = _run_library(
            compute_alpha101, ALPHA101_NAMES, frames,
            {"cap": cap, "industry": industry}, CHUNK_101, "alpha101",
            horizons, ret_ranks, args.min_pairs, args.min_dates)
        rows += r101
        series.update(s101)
        libs.append("alpha101")
    if args.only in ("158", "both"):
        r158, s158 = _run_library(
            compute_alpha158, ALPHA158_NAMES, frames, {}, CHUNK_158,
            "alpha158", horizons, ret_ranks, args.min_pairs, args.min_dates)
        rows += r158
        series.update(s158)
        libs.append("alpha158")

    _save_summaries(rows, libs, horizons, outdir)
    for lib in libs:
        _plot_ranking(rows, lib, horizons[0], outdir)
    _save_top_series(series, rows, horizons[0], outdir)

    # 覆盖率小结
    n_reported = len({(r["factor"], r["lib"]) for r in rows})
    print(f"[ok] 汇总: {n_reported} 个因子有足够 IC 历史；输出目录 {outdir}/")
    print(DISCLAIMER)
    return 0


# ---------------------------------------------------------------------------
# 自检（合成数据，不访问网络）
# ---------------------------------------------------------------------------

def _selftest() -> int:
    rng = np.random.default_rng(43)
    n_days, n_stocks = 60, 8
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    syms = [f"S{i:02d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)),
        index=dates, columns=syms)
    open_ = close * (1 + rng.normal(0, 0.005, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, close.shape)))
    volume = pd.DataFrame(rng.integers(10_000, 100_000, close.shape),
                          index=dates, columns=syms).astype(float)
    frames = {"open": open_, "high": high, "low": low, "close": close,
              "volume": volume, "vwap": close * 1.001,
              "returns": close.pct_change()}

    # 1. 未来收益口径：最后一交易日全 NaN，逐列手算一致
    ret = forward_returns(close, 1)
    assert ret.iloc[-1].isna().all(), "最后一交易日应有 NaN 未来收益"
    for s in syms:
        manual = (close[s].shift(-1) / close[s] - 1.0)
        assert np.allclose(ret[s].dropna(), manual.dropna()), \
            f"{s} 未来 1 日收益与手算不符"
    ret5 = forward_returns(close, 5)
    assert ret5.iloc[-5:].isna().all().all(), "最后 5 日应全 NaN"
    print("[ok] 未来收益：口径与手算一致，尾部 NaN 正确")

    # 2. IC 恒等式：因子=未来收益 → IC≡1；因子=-未来收益 → IC≡-1
    r_rank = _rank_matrix(ret)
    ic_self = cross_sectional_ic(ret, r_rank, min_pairs=3)
    assert np.allclose(ic_self.dropna(), 1.0), "因子=未来收益时 IC 应为 1"
    ic_neg = cross_sectional_ic(-ret, r_rank, min_pairs=3)
    assert np.allclose(ic_neg.dropna(), -1.0), "因子=-未来收益时 IC 应为 -1"
    print("[ok] IC 恒等式：自身/取负未来收益的 IC 为 ±1")

    # 3. 常数因子（无横截面方差）→ IC 全 NaN，不崩溃
    const = pd.DataFrame(1.0, index=dates, columns=syms)
    ic_const = cross_sectional_ic(const, r_rank, min_pairs=3)
    assert ic_const.isna().all(), "常数因子的 IC 应为 NaN"
    print("[ok] 常数因子 IC 全 NaN，无除零崩溃")

    # 4. 最小配对门槛：只剩 2 只有效时（<min_pairs=3）IC 置 NaN
    sparse = ret.copy()
    sparse.iloc[:, 2:] = np.nan
    ic_sparse = cross_sectional_ic(sparse, r_rank, min_pairs=3)
    assert ic_sparse.isna().all(), "低于最小配对门槛应置 NaN"
    print("[ok] 最小配对门槛生效")

    # 5. 与 pandas corr 逐日对照：无 NaN 的随机因子 + NaN 模式与收益
    #    不同的因子（两侧去均值/求和都必须取交集，回归用例）
    rng5 = np.random.default_rng(5)
    noise = pd.DataFrame(rng5.normal(size=close.shape), index=dates,
                         columns=syms)
    ragged = noise.copy()
    ragged.iloc[::7, 1::3] = np.nan  # 因子 NaN 模式 ≠ 收益 NaN 模式
    for name, frame in (("noise", noise), ("ragged", ragged)):
        f_rank = frame.rank(axis=1)
        ic_ref = f_rank.corrwith(pd.DataFrame(r_rank, index=dates,
                                              columns=syms),
                                 axis=1, method="pearson")
        ic_ours = cross_sectional_ic(frame, r_rank, min_pairs=3)
        assert np.allclose(ic_ours.dropna(), ic_ref.dropna(), atol=1e-12), \
            f"{name}: 向量化 IC 与 pandas 逐日秩相关不一致"
    print("[ok] 向量化 IC 与 pandas 逐日秩相关一致（含 NaN 模式差异用例）")

    # 6. 与因子库联动：alpha001 在合成数据上可计算且 IC 有限/NaN 不崩溃
    a001 = compute_alpha101(frames, include=["alpha001"])["alpha001"]
    ic_a001 = cross_sectional_ic(a001, r_rank, min_pairs=3)
    assert ic_a001.notna().any(), "alpha001 应有有效 IC 日期"
    print("[ok] 因子库联动：alpha001 IC 序列可计算")

    print("== factor_ic selftest 全部通过 ==")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="259 个开源因子的 A 股有效性体检（IC 筛选，issue #43）",
        epilog="示例: python scripts/factor_ic.py run --max-stocks 1200")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="合成数据自检（不访问网络）")

    p = sub.add_parser("run", help="TSP 导入 + 因子计算 + IC 汇总")
    p.add_argument("--tsp-dir", default=TSP_DIR_DEFAULT,
                   help="本机 TSP 数据目录")
    p.add_argument("--start", default="20190719", help="开始日期 YYYYMMDD")
    p.add_argument("--end", default="20261231", help="结束日期 YYYYMMDD")
    p.add_argument("--max-stocks", type=int, default=1200,
                   help="股票池上限（等距抽样；None 不限制）")
    p.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT,
                   help="单股最少交易日数（股票池过滤，约 1 年）")
    p.add_argument("--min-pairs", type=int, default=MIN_PAIRS_DEFAULT,
                   help="单日横截面最少有效配对")
    p.add_argument("--min-dates", type=int, default=MIN_IC_DATES_DEFAULT,
                   help="因子汇总最少有效 IC 日期数")
    p.add_argument("--horizons", type=str, default="1,5",
                   help="未来收益口径（逗号分隔的交易日数）")
    p.add_argument("--only", choices=["101", "158", "both"], default="both")
    p.add_argument("--no-fetch", action="store_true",
                   help="跳过 TSP 导入，直接复用 --datadir 下已有 CSV")
    p.add_argument("--datadir", default=DATADIR_DEFAULT, help="日线 CSV 目录")
    p.add_argument("--outdir", default=OUTDIR_DEFAULT, help="输出目录")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "selftest":
        return _selftest()
    args.horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if not args.horizons:
        print("--horizons 至少需要一个交易日数", file=sys.stderr)
        return 1
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
