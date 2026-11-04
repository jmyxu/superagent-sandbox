"""申万一级行业动量轮动策略（量化学习项目第 8 步：A 股特色策略）。

策略（月度行业动量排序选 Top-N 行业、行业内等权）：
  - 每月最后一个交易日收盘，用申万一级行业指数过去 --ind-lookback 个交易日
    的累计收益（可跳过最近 --ind-skip 个交易日）对 31 个行业排序；
  - 动量方向选收益最大的 Top-K 行业；反转方向（--direction reversal）选
    收益最小的 Top-K 行业（A 股 1 个月窗口的反转效应强于动量，见调研）；
  - 行业间等权（每个入选行业 1/K），行业内对股票池中该行业的股票等权；
  - t 日收盘信号 → t+1 日开盘成交（无未来函数），停牌/未上市跳过、份额留
    现金，成本口径与 scripts/screener.py 完全一致（佣金万三 + 印花万五）。

数据：
  - 行业指数：akshare 申万宏源研究接口 `index_hist_sw`（31 个一级行业指数
    日线，1999-12-30 起），缓存于 data/sw/sw_hist/{code}.csv；
  - 行业成分：akshare `index_component_sw`（申万一级行业**当前**成分股
    快照），缓存于 data/sw/sw_cons/{code}.csv；
  - 股票日线：复用第 6 步的 data/daily 缓存（沪深300+中证500 当前成分股，
    约 800 只，qfq）。

已知局限（延续第 6 步口径并如实声明）：
  1. 行业成分是**当前**快照——历史上换过行业的股票按今天的行业归属参与
     全部回测（point-in-time 行业分类数据 `stock_industry_clf_hist_sw`
     存在但上游不稳定，留作后续扩展）；
  2. 股票池仍为当前成分股快照，幸存者偏差与第 6 步相同；
  3. 只有股票池内（约 800 只）的股票可被执行，"行业内等权"指池内该行业
     股票等权；行业指数本身仍用于全 31 行业排序。

策略来源与调研（详见 docs/optimization-log.md 第 11 章）：
  - 聚宽社区"行业轮动：基于申万行业指数的动量策略"：
    https://easyquant.ai/e/joinquant/sw-industry-rotation-strategy
  - BigQuant《行业轮动策略（优化版）》：
    https://wiki2.bigquant.com/wiki/doc/oGZs02a2Lf
  - 聚宽社区"行业轮动的黄金律：日内动量与隔夜反转"：
    https://www.joinquant.com/view/community/detail/770646f27ea44a0076ae9113316c7f23
  - Moskowitz & Grinblatt (1999), J. Finance 54(4) 1249-1290（行业动量
    的学术源头）

用法：
    # 自检（合成数据，不访问网络）
    python scripts/industry_rotation.py selftest

    # 1) 拉取申万一级行业列表/指数日线/成分股（增量缓存）
    python scripts/industry_rotation.py fetch

    # 2) 月度行业动量轮动回测（默认 Top-5、20 日动量）
    python scripts/industry_rotation.py backtest
    python scripts/industry_rotation.py backtest --direction reversal

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证以 `python scripts/industry_rotation.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backtest import (  # noqa: E402
    DISCLAIMER,
    _setup_cjk_font,
    compute_metrics,
    standardize_stock,
)
from screener import (  # noqa: E402
    PANEL_START,
    _load_benchmark_cached,
    build_panel,
    fetch_universe,
    rebalance_dates,
)

BACKTEST_START_DEFAULT = "20200101"
BACKTEST_END_DEFAULT = "20251231"

SW_DIR = "data/sw"
SW_INDUSTRIES_CSV = "sw_industries.csv"

# 申万一级行业指数日线列名（akshare index_hist_sw）
_HIST_RENAME = {"日期": "date", "收盘": "close"}


# ---------------------------------------------------------------------------
# 申万行业数据：拉取 + 缓存
# ---------------------------------------------------------------------------

def fetch_sw_industries(offline: bool = False, retries: int = 3
                        ) -> pd.DataFrame:
    """申万一级行业列表（31 个）：code / name，缓存 data/sw/sw_industries.csv。"""
    import akshare as ak

    fdir = Path(SW_DIR)
    fdir.mkdir(parents=True, exist_ok=True)
    cache_path = fdir / SW_INDUSTRIES_CSV
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"code": str})
        if not df.empty:
            return df
    if offline:
        raise RuntimeError(f"offline 模式且无行业列表缓存（{cache_path}）")

    raw = None
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = ak.sw_index_first_info()
            if raw is not None and not raw.empty:
                break
            last_err = ValueError("接口返回空行业列表")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    if raw is None or raw.empty:
        raise RuntimeError(f"申万一级行业列表拉取失败: {last_err}")
    df = pd.DataFrame({
        "code": raw["行业代码"].astype(str).str.replace(".SI", "",
                                                       regex=False),
        "name": raw["行业名称"],
    }).drop_duplicates("code").sort_values("code").reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[ok] 申万一级行业列表: {len(df)} 个，已缓存 {cache_path}")
    return df


def fetch_sw_hist(industries: pd.DataFrame, offline: bool = False,
                  retries: int = 3) -> None:
    """31 个行业指数日线（1999-12-30 起全历史），缓存 data/sw/sw_hist/。"""
    import akshare as ak

    outdir = Path(SW_DIR) / "sw_hist"
    outdir.mkdir(parents=True, exist_ok=True)
    ok = cached = 0
    for _, row in industries.iterrows():
        code, name = row["code"], row["name"]
        cache_path = outdir / f"{code}.csv"
        if cache_path.exists():
            try:
                d = pd.read_csv(cache_path)
                fresh_enough = pd.Timestamp.now() - pd.Timedelta(days=14)
                if not d.empty and pd.to_datetime(d["date"]).max() >= \
                        fresh_enough:
                    cached += 1
                    continue
            except Exception:  # noqa: BLE001 损坏文件重拉
                pass
        if offline:
            raise RuntimeError(f"offline 模式且无行业指数缓存（{cache_path}）")
        last_err: Exception | None = None
        df = None
        for attempt in range(1, retries + 1):
            try:
                df = ak.index_hist_sw(symbol=code, period="day")
                if df is not None and not df.empty:
                    break
                last_err = ValueError("接口返回空数据")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries:
                    time.sleep(2 * attempt)
        if df is None or df.empty:
            raise RuntimeError(f"{code} {name}: 行业指数拉取失败: {last_err}")
        out = df.rename(columns=_HIST_RENAME)[["date", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
        ok += 1
        if (ok + cached) % 10 == 0:
            print(f"[进度] 行业指数 {ok + cached}/{len(industries)}")
    print(f"[ok] 行业指数: 新拉 {ok}、缓存 {cached}")


def fetch_sw_cons(industries: pd.DataFrame, offline: bool = False,
                  retries: int = 3) -> None:
    """31 个行业的当前成分股快照，缓存 data/sw/sw_cons/{code}.csv。"""
    import akshare as ak

    outdir = Path(SW_DIR) / "sw_cons"
    outdir.mkdir(parents=True, exist_ok=True)
    ok = cached = 0
    for _, row in industries.iterrows():
        code, name = row["code"], row["name"]
        cache_path = outdir / f"{code}.csv"
        if cache_path.exists():
            try:
                d = pd.read_csv(cache_path, dtype={"symbol": str})
                if not d.empty:
                    cached += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        if offline:
            raise RuntimeError(f"offline 模式且无成分股缓存（{cache_path}）")
        last_err: Exception | None = None
        df = None
        for attempt in range(1, retries + 1):
            try:
                df = ak.index_component_sw(symbol=code)
                if df is not None and not df.empty:
                    break
                last_err = ValueError("接口返回空数据")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries:
                    time.sleep(2 * attempt)
        if df is None or df.empty:
            raise RuntimeError(f"{code} {name}: 成分股拉取失败: {last_err}")
        out = pd.DataFrame({
            "symbol": df["证券代码"].astype(str).str.zfill(6),
            "name": df["证券名称"],
        }).drop_duplicates("symbol")
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
        ok += 1
        if (ok + cached) % 10 == 0:
            print(f"[进度] 行业成分 {ok + cached}/{len(industries)}")
    print(f"[ok] 行业成分: 新拉 {ok}、缓存 {cached}")


def load_sw_data(offline: bool, retries: int
                 ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame],
                            dict[str, str]]:
    """加载（必要时拉取）行业列表、指数日线、成分股。

    返回 (industries, hist: code→date/close DataFrame, stock_industry:
    symbol→行业代码)。成分股为当前快照：symbol 跨行业重复时取最新权重
    更大者（直接丢弃重复行即可，接口按行业返回，重复概率极低）。
    """
    industries = fetch_sw_industries(offline, retries)
    if not (Path(SW_DIR) / "sw_hist").exists() or not any(
            (Path(SW_DIR) / "sw_hist").glob("*.csv")):
        fetch_sw_hist(industries, offline, retries)
    if not (Path(SW_DIR) / "sw_cons").exists() or not any(
            (Path(SW_DIR) / "sw_cons").glob("*.csv")):
        fetch_sw_cons(industries, offline, retries)

    hist: dict[str, pd.DataFrame] = {}
    stock_industry: dict[str, str] = {}
    for _, row in industries.iterrows():
        code = row["code"]
        h = pd.read_csv(Path(SW_DIR) / "sw_hist" / f"{code}.csv")
        h["date"] = pd.to_datetime(h["date"])
        h = h.sort_values("date").drop_duplicates("date", keep="last")
        hist[code] = h.set_index("date")["close"]
        c = pd.read_csv(Path(SW_DIR) / "sw_cons" / f"{code}.csv",
                        dtype={"symbol": str})
        for sym in c["symbol"]:
            stock_industry.setdefault(sym, code)
    return industries, hist, stock_industry


# ---------------------------------------------------------------------------
# 行业动量因子与组合引擎
# ---------------------------------------------------------------------------

def industry_momentum(idx_close: np.ndarray, t: int, lookback: int,
                      skip: int) -> np.ndarray:
    """行业动量：指数 close[t-skip]/close[t-lookback]-1。

    idx_close 为 n_dates × n_industries（行业指数收盘价，按并集日历对齐、
    指数自身数据范围内前向填充）。只用 <= t 的信息；数据不足的行业返回
    NaN（不参与排序）。
    """
    n_ind = idx_close.shape[1]
    out = np.full(n_ind, np.nan)
    a, b = t - skip, t - lookback
    if a < 0 or b < 0:
        return out
    pa, pb = idx_close[a], idx_close[b]
    valid = ~np.isnan(pa) & ~np.isnan(pb) & (pa > 0) & (pb > 0)
    out[valid] = pa[valid] / pb[valid] - 1.0
    return out


def select_top_industries(values: np.ndarray, k: int, direction: int
                          ) -> list[int]:
    """按因子值选 Top-K 个行业列下标。direction=+1 选最大（动量），
    -1 选最小（反转）。有效行业不足 K 个时选全部有效。"""
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return []
    order = valid[np.argsort(-direction * values[valid])]
    return order[:k].tolist()


def plan_weights(selected: list[int], industry_of: np.ndarray,
                 tradable: list[int]) -> dict[int, list[int]]:
    """把入选行业映射到可交易股票：{行业列下标: [股票列下标]}。

    行业间等权在引擎里按 1/n_eff 实现（n_eff = 有可交易股票的入选行业数），
    行业内按 1/len(组内股票) 实现。
    """
    groups: dict[int, list[int]] = {}
    for j in tradable:
        ind = industry_of[j]
        if ind in selected:
            groups.setdefault(ind, []).append(j)
    return groups


def run_backtest(dates: pd.DatetimeIndex, close: pd.DataFrame,
                 open_: pd.DataFrame, symbols: list[str],
                 industry_of: np.ndarray, industry_codes: list[str],
                 idx_close: pd.DataFrame, args: argparse.Namespace,
                 benchmark: pd.DataFrame, names: dict[str, str],
                 ind_names: dict[str, str]) -> dict:
    """月度行业轮动回测：行业指数排序 → Top-K 行业 → 池内等权（行业间
    等权、行业内等权），t+1 开盘成交。口径与 screener.run_backtest 一致。
    """
    close_np = close.to_numpy(dtype=float)
    open_np = open_.to_numpy(dtype=float)
    idx_np = idx_close.to_numpy(dtype=float)
    n, m = close_np.shape
    start = pd.Timestamp(args.start)
    eval_start = int(np.searchsorted(dates, start))
    dates_ev = dates[eval_start:]
    if len(dates_ev) == 0:
        raise RuntimeError(f"回测起点 {args.start} 之后无交易日")
    n_ev = len(dates_ev)
    years = max((dates_ev[-1] - dates_ev[0]).days / 365.25, 1e-9)

    bench_s = benchmark.set_index("date")["close"].sort_index()
    bench_aligned = bench_s.reindex(dates).ffill().bfill().to_numpy(float)
    bench_curve = bench_aligned[eval_start:] / bench_aligned[eval_start]

    reb_dates = rebalance_dates(dates_ev)
    reb_idx = [int(np.searchsorted(dates, d)) for d in reb_dates]

    def _run_one(direction: int) -> dict:
        capital = float(args.capital)
        cash = capital
        shares = np.zeros(m)
        equity = np.full(n_ev, np.nan)
        equity[0] = capital
        turnover = np.zeros(n_ev)
        trade_days = 0
        holdings: list[dict] = []
        latest: list[dict] = []
        next_fill = 1

        for t in reb_idx:
            k = t + 1
            if k >= n:
                break
            k_ev = k - eval_start
            for kk_ev in range(next_fill, k_ev):
                kk = eval_start + kk_ev
                equity[kk_ev] = cash + float(np.nansum(shares * close_np[kk]))

            vals = industry_momentum(idx_np, t, args.ind_lookback,
                                     args.ind_skip)
            selected = select_top_industries(vals, args.top_k, direction)
            t_open = open_np[k]
            tradable = [j for j in range(m)
                        if not np.isnan(t_open[j]) and t_open[j] > 0
                        and not np.isnan(close_np[t, j])]
            groups = plan_weights(selected, industry_of, tradable)
            sel_names = [ind_names.get(industry_codes[i],
                                       industry_codes[i])
                         for i in selected]
            latest = [{
                "股票代码": symbols[j],
                "股票名称": (names or {}).get(symbols[j], ""),
                "行业": ind_names.get(industry_codes[industry_of[j]],
                                     industry_codes[industry_of[j]]),
            } for j in sorted({j for grp in groups.values() for j in grp})]
            holdings.append({
                "date": dates_ev[k_ev - 1].strftime("%Y-%m-%d"),
                "industries": " ".join(sel_names),
                "n_industries": len(groups),
                "n_holdings": sum(len(v) for v in groups.values()),
                "symbols": " ".join(symbols[j] for grp in groups.values()
                                    for j in sorted(grp)),
            })
            if not groups:
                equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
                next_fill = k_ev + 1
                continue

            eff_open = np.where(np.isnan(t_open), close_np[t], t_open)
            nav_before = cash + float(np.nansum(shares * eff_open))
            n_eff = len(groups)
            trades_value = 0.0
            trade_days += 1
            held = {j for grp in groups.values() for j in grp}

            for j in np.flatnonzero(shares > 0):
                if j not in held and not np.isnan(t_open[j]):
                    sell_value = float(shares[j] * t_open[j])
                    fee = sell_value * (args.commission + args.stamp)
                    cash += sell_value - fee
                    trades_value += sell_value
                    shares[j] = 0.0

            for grp in groups.values():
                w_stock = 1.0 / (n_eff * len(grp))
                for j in grp:
                    target_value = nav_before * w_stock
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
            equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
            next_fill = k_ev + 1

        for kk_ev in range(next_fill, n_ev):
            kk = eval_start + kk_ev
            equity[kk_ev] = cash + float(np.nansum(shares * close_np[kk]))
        equity = pd.Series(equity).ffill().to_numpy()
        return {
            "equity": equity,
            "turnover_annual": float(np.sum(turnover) / years),
            "trade_days": trade_days,
            "holdings": pd.DataFrame(holdings),
            "n_holdings_mean": float(np.mean(
                [h["n_holdings"] for h in holdings])) if holdings else np.nan,
            "latest": latest,
        }

    results: dict[str, dict] = {}
    for direction, label in ((+1, "momentum"), (-1, "reversal")):
        if args.direction in (label, "both"):
            results[label] = _run_one(direction)

    # 等权池基准（与 screener 同口径：月度再平衡、同成本）
    capital = float(args.capital)
    cash = capital
    shares = np.zeros(m)
    ew_equity = np.full(n_ev, np.nan)
    ew_equity[0] = capital
    next_fill = 1
    for t in reb_idx:
        k = t + 1
        if k >= n:
            break
        k_ev = k - eval_start
        for kk_ev in range(next_fill, k_ev):
            kk = eval_start + kk_ev
            ew_equity[kk_ev] = cash + float(np.nansum(shares * close_np[kk]))
        t_open = open_np[k]
        t_close = close_np[t]
        tradable = [j for j in range(m)
                    if not np.isnan(t_open[j]) and not np.isnan(t_close[j])
                    and t_open[j] > 0]
        if not tradable:
            ew_equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
            next_fill = k_ev + 1
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
        ew_equity[k_ev] = cash + float(np.nansum(shares * close_np[k]))
        next_fill = k_ev + 1
    for kk_ev in range(next_fill, n_ev):
        kk = eval_start + kk_ev
        ew_equity[kk_ev] = cash + float(np.nansum(shares * close_np[kk]))
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

ROTATION_LABELS = {"momentum": "行业动量 Top-K", "reversal": "行业反转 Top-K"}


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def write_outputs(out: dict, args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = out["dates"]

    curves = pd.DataFrame({"date": dates})
    metrics_rows: list[dict] = []
    for label, res in out["results"].items():
        disp = ROTATION_LABELS[label]
        m = compute_metrics(res["equity"], 0)
        curves[f"{label}_nav"] = res["equity"] / res["equity"][0]
        metrics_rows.append({
            "series": disp, "type": "strategy",
            "total": m["total"], "annual": m["annual"], "mdd": m["mdd"],
            "turnover_annual": res["turnover_annual"],
            "trade_days": res["trade_days"],
            "n_holdings_mean": res["n_holdings_mean"],
        })
        if not res["holdings"].empty:
            res["holdings"].to_csv(
                outdir / f"holdings_{label}.csv",
                index=False, encoding="utf-8-sig")
        latest = res.get("latest")
        if latest:
            pd.DataFrame(latest).to_csv(
                outdir / f"latest_picks_{label}.csv",
                index=False, encoding="utf-8-sig")
            print(f"\n最新一期 {disp}（{len(latest)} 只，行业内等权清单）:")
            for i, row in enumerate(latest[:15], 1):
                print(f"  {i:>2}. {row['股票代码']} {row['股票名称']}"
                      f"（{row['行业']}）")
            if len(latest) > 15:
                print(f"  … 共 {len(latest)} 只，完整清单见 "
                      f"{outdir / f'latest_picks_{label}.csv'}")

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
    summary.to_csv(outdir / "summary_sw_rotation.csv", index=False,
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
    print(f"\n===== 申万一级行业轮动 · {args.start} ~ {args.end}，"
          f"Top-K={args.top_k}，{args.ind_lookback} 日动量跳过 "
          f"{args.ind_skip} 日，qfq，初始 {args.capital:,.0f} 元 =====")
    print(pd.DataFrame(table).to_string())

    if not args.no_plot:
        _setup_cjk_font()
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={
                                           "height_ratios": [3, 1]})
        colors = {"momentum": "tab:blue", "reversal": "tab:purple"}
        for label, res in out["results"].items():
            nav = curves[f"{label}_nav"]
            ax1.plot(dates, nav, lw=1.4, color=colors[label],
                     label=ROTATION_LABELS[label])
            dd = nav / nav.cummax() - 1
            ax2.plot(dates, dd, lw=0.8, color=colors[label], alpha=0.7)
        ax1.plot(dates, curves["ew_pool_nav"], lw=1.1, color="tab:orange",
                 alpha=0.75, label="等权股票池基准")
        ax1.plot(dates, curves["csi300_nav"], lw=1.1, color="tab:green",
                 alpha=0.75, label="沪深300基准")
        ax1.set_title(f"申万一级行业轮动 · 月度 Top-{args.top_k}"
                      f"（{args.start} ~ {args.end}，qfq）")
        ax1.set_ylabel("净值（期初=1）")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)
        ax2.set_ylabel("回撤")
        ax2.grid(alpha=0.3)
        fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=9,
                 color="gray")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        out_png = outdir / "sw_rotation_compare.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[ok] 对比图已保存: {out_png}")


# ---------------------------------------------------------------------------
# 自检（合成数据，不访问网络）
# ---------------------------------------------------------------------------

def self_test() -> None:
    # 1) 行业动量计算：t=250 时 idx0 上行、idx1 下行、idx2 数据不足
    days = 400
    rng = np.random.default_rng(7)
    base = np.arange(days)
    idx = np.full((days, 3), np.nan)
    idx[:, 0] = 100.0 * np.cumprod(1.0 + 0.001 + 0.0005 * rng.standard_normal(days))
    idx[:, 1] = 100.0 * np.cumprod(1.0 - 0.001 + 0.0005 * rng.standard_normal(days))
    idx[100:, 2] = 100.0 * np.cumprod(1.0 + 0.001 + 0.0005 * rng.standard_normal(days - 100))
    mom = industry_momentum(idx, 250, lookback=20, skip=0)
    assert mom[0] > 0 and mom[1] < 0, f"行业动量方向错误: {mom}"
    assert not np.isnan(mom[2]), "idx2 在 t=250 已有 150 根，应有效"
    mom_early = industry_momentum(idx, 50, lookback=20, skip=0)
    assert np.isnan(mom_early[2]), "idx2 在 t=50 无数据，应为 NaN"
    # 无未来函数：改 t+1 之后的数据不影响 t 时点因子
    tampered = idx.copy()
    tampered[251:] = 999.0
    mom_t = industry_momentum(tampered, 250, lookback=20, skip=0)
    assert np.allclose(mom_t, mom, equal_nan=True), "行业动量使用了未来数据"

    # 2) Top-K 选择方向
    vals = np.array([0.1, -0.2, np.nan, 0.05])
    top2_mom = select_top_industries(vals, 2, direction=+1)
    assert top2_mom == [0, 3], f"行业动量 Top-2 应为 [0,3]，实际 {top2_mom}"
    top2_rev = select_top_industries(vals, 2, direction=-1)
    assert top2_rev == [1, 3], f"行业反转 Top-2 应为 [1,3]，实际 {top2_rev}"

    # 3) 权重规划：行业间等权、行业内等权（引擎以 1/n_eff × 1/len 实现）
    industry_of = np.array([0, 0, 1, 1, -1])  # 股票 j → 行业列下标；-1=未知
    groups = plan_weights([0, 1], industry_of, [0, 1, 2, 3, 4])
    assert groups == {0: [0, 1], 1: [2, 3]}, \
        f"权重分组错误（未知行业应排除）: {groups}"
    groups_sub = plan_weights([0], industry_of, [0, 2, 4])
    assert groups_sub == {0: [0]}, f"未入选行业不应建仓: {groups_sub}"

    # 4) 迷你回测手工验算：2 行业 × 2 股票，行业间等权、行业内等权
    d2 = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-28",
                           "2020-03-02"])
    c2 = pd.DataFrame({"s0": [10.0, 11.0, 12.0, 13.0],
                       "s1": [20.0, 10.0, 30.0, 15.0]}, index=d2)
    o2 = pd.DataFrame({"s0": [10.0, 11.0, 12.0, 12.5],
                       "s1": [20.0, 10.0, 30.0, 21.0]}, index=d2)
    # 行业指数：indA 上行（t=2 时 20/10-1=+100%）、indB 下行（-40%）
    idx2 = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0],
                         "B": [10.0, 10.0, 6.0, 6.0]}, index=d2)
    industry_of = np.array([0, 1])  # s0→A, s1→B
    bench = pd.DataFrame({"date": d2, "close": [1.0, 1.0, 1.0, 1.0]})
    mini = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0, stamp=0.0, top_k=1, ind_lookback=2, ind_skip=0,
        direction="momentum")
    out = run_backtest(d2, c2, o2, ["s0", "s1"], industry_of, ["A", "B"],
                       idx2, mini, bench, {}, {"A": "行业A", "B": "行业B"})
    res = out["results"]["momentum"]
    # Top-1 行业 = A（动量 +100%）→ 全仓 s0：10000/12.5*13 = 10400
    assert abs(res["equity"][-1] - 10400.0) < 0.01, \
        f"行业动量迷你回测 {res['equity'][-1]} 与手算 10400 不符"
    assert res["holdings"]["industries"].iloc[1] == "行业A", \
        "Top-1 应选行业A"
    assert res["holdings"]["symbols"].iloc[1] == "s0", \
        "行业内应持有 s0"
    # 方向反转：Top-1 = B（-40%）→ 全仓 s1：10000/21*15 = 7142.857
    mini_rev = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0, stamp=0.0, top_k=1, ind_lookback=2, ind_skip=0,
        direction="reversal")
    out_rev = run_backtest(d2, c2, o2, ["s0", "s1"], industry_of, ["A", "B"],
                           idx2, mini_rev, bench, {}, {"A": "行业A",
                                                       "B": "行业B"})
    res_rev = out_rev["results"]["reversal"]
    expected_rev = 10000.0 / 21.0 * 15.0
    assert abs(res_rev["equity"][-1] - expected_rev) < 0.01, \
        f"行业反转迷你回测 {res_rev['equity'][-1]} 与手算 {expected_rev} 不符"

    # 5) 行业间等权：Top-2 两行业各 1 只股票 → 各 50% 资金（零成本）
    mini2 = argparse.Namespace(
        start="2020-01-31", end="2020-03-02", capital=10000.0,
        commission=0.0, stamp=0.0, top_k=2, ind_lookback=2, ind_skip=0,
        direction="momentum")
    out2 = run_backtest(d2, c2, o2, ["s0", "s1"], industry_of, ["A", "B"],
                        idx2, mini2, bench, {}, {"A": "行业A", "B": "行业B"})
    res2 = out2["results"]["momentum"]
    # 各 50%：s0 4000/12.5 股、s1 5000/21 股 → 期末 4000/12.5*13 + 5000/21*15
    expected2 = (5000.0 / 12.5) * 13.0 + (5000.0 / 21.0) * 15.0
    assert abs(res2["equity"][-1] - expected2) < 0.01, \
        f"行业间等权迷你回测 {res2['equity'][-1]} 与手算 {expected2} 不符"

    print("[ok] self_test 通过：行业动量计算、无未来函数、Top-K 方向、"
          "权重规划、迷你回测与行业间/行业内等权断言全部成立")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="申万一级行业动量轮动：月度行业指数动量排序选 Top-K 行业，"
                    "行业内等权（量化学习项目第 8 步）",
        epilog=(
            "示例:\n"
            "  python scripts/industry_rotation.py fetch\n"
            "  python scripts/industry_rotation.py backtest\n"
            "  python scripts/industry_rotation.py backtest --direction "
            "reversal --top-k 3\n"
            "  python scripts/industry_rotation.py selftest"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("fetch", help="拉取申万一级行业列表/指数日线/成分股"
                                 "（增量缓存）")

    p_bt = sub.add_parser("backtest", help="行业动量轮动回测")
    p_bt.add_argument("--start", default=BACKTEST_START_DEFAULT,
                      help=f"回测开始日期 YYYYMMDD，默认 "
                           f"{BACKTEST_START_DEFAULT}")
    p_bt.add_argument("--end", default=BACKTEST_END_DEFAULT,
                      help=f"回测结束日期 YYYYMMDD，默认 {BACKTEST_END_DEFAULT}")
    p_bt.add_argument("--top-k", type=int, default=5,
                      help="每月选入的行业数 Top-K，默认 5")
    p_bt.add_argument("--ind-lookback", type=int, default=20,
                      help="行业动量回看窗口（交易日），默认 20（约 1 个月）")
    p_bt.add_argument("--ind-skip", type=int, default=0,
                      help="行业动量跳过的最近交易日数，默认 0")
    p_bt.add_argument("--direction", choices=["momentum", "reversal", "both"],
                      default="momentum",
                      help="momentum=选收益最大行业（默认）；reversal=选收益"
                           "最小行业；both=两者都跑")
    p_bt.add_argument("--datadir", default="data/daily",
                      help="股票日线 CSV 目录，默认 data/daily")
    p_bt.add_argument("--outdir", default="output/industry_rotation",
                      help="输出目录，默认 output/industry_rotation")
    p_bt.add_argument("--capital", type=float, default=100000.0,
                      help="初始资金（元），默认 100000")
    p_bt.add_argument("--commission", type=float, default=0.0003,
                      help="单边佣金费率，默认 0.0003（万三）")
    p_bt.add_argument("--stamp", type=float, default=0.0005,
                      help="卖出印花税费率，默认 0.0005（万五）")
    p_bt.add_argument("--retries", type=int, default=3,
                      help="数据拉取失败重试次数，默认 3")
    p_bt.add_argument("--offline", action="store_true",
                      help="只用本地 CSV，不访问网络")
    p_bt.add_argument("--no-plot", action="store_true", help="不生成 PNG 对比图")

    sub.add_parser("selftest", help="只运行合成数据上的算法自检（不拉数据）")

    args = parser.parse_args(argv)
    if args.command == "backtest":
        if args.top_k <= 0:
            parser.error("--top-k 必须为正整数")
        if args.ind_lookback <= args.ind_skip:
            parser.error("--ind-lookback 必须大于 --ind-skip")
    return args


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

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(90)
    try:
        if command == "fetch":
            industries = fetch_sw_industries(offline=False,
                                             retries=getattr(args, "retries",
                                                             3))
            fetch_sw_hist(industries, offline=False, retries=3)
            fetch_sw_cons(industries, offline=False, retries=3)
            return 0

        # backtest
        industries, hist, stock_industry = load_sw_data(args.offline,
                                                        args.retries)
        universe = fetch_universe(offline=args.offline, retries=args.retries)
        symbols = universe["symbol"].tolist()
        names = dict(zip(universe["symbol"], universe["name"]))
        industry_codes = industries["code"].tolist()
        ind_names = dict(zip(industries["code"], industries["name"]))
        code_pos = {c: i for i, c in enumerate(industry_codes)}
        industry_of = np.array(
            [code_pos.get(stock_industry.get(s, ""), -1) for s in symbols])

        # 行业指数面板：各指数自 1999-12-30 起，按并集日历对齐（前向填充）
        idx_close = pd.DataFrame({c: hist[c] for c in industry_codes})
        idx_close = idx_close.ffill()

        n_mapped = int(np.sum(industry_of >= 0))
        print(f"[ok] 行业列表 {len(industry_codes)} 个；股票池 {len(symbols)} "
              f"只中 {n_mapped} 只映射到申万一级行业")

        dates, close, open_ = build_panel(symbols, args.datadir, "qfq",
                                          PANEL_START, args.end)
        if len(dates) == 0 or close.empty:
            print("[error] 面板为空：请先运行 fetch 拉取日线", file=sys.stderr)
            return 1
        print(f"[ok] 面板: {len(dates)} 个交易日 × {close.shape[1]} 只股票")
        idx_close = idx_close.reindex(dates).ffill()
        if idx_close.isna().all().all():
            print("[error] 行业指数面板为空", file=sys.stderr)
            return 1

        bench, bench_source = _load_benchmark_cached(args, "data/daily")
        print(f"[ok] 沪深300基准: {bench_source}，{len(bench)} 行")

        out = run_backtest(dates, close, open_, symbols, industry_of,
                           industry_codes, idx_close, args, bench, names,
                           ind_names)
        write_outputs(out, args)
        print(f"\n⚠ {DISCLAIMER}")
        return 0
    finally:
        socket.setdefaulttimeout(old_timeout)


if __name__ == "__main__":
    sys.exit(main())
