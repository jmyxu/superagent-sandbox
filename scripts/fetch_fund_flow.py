"""东财免费口径：个股资金流历史拉取（issue #34，大单资金流因子研究）。

数据源：东方财富免费接口 push2his.eastmoney.com .../fflow/daykline/get。
该接口按 东财资金流向口径 返回每只股票每日的 主力/超大单/大单/中单/小单
净流入（净额与净占比）。⚠ 免费接口只提供**最近约 120 个交易日**的历史
（截至研究日 2026-11-13 约覆盖 2026-05-21 起），更长历史需东财付费口径——
这是本研究回测窗口的上限，见 docs/fund-flow-factor-research.md。

东财口径（每笔成交按金额划分，净流入 = 主动买入额 - 主动卖出额）：
- 超大单：单笔 ≥ 100 万元（或 ≥ 50 万股）
- 大单：20 万 ~ 100 万元
- 中单：4 万 ~ 20 万元
- 小单：< 4 万元
- 主力 = 超大单 + 大单

本脚本复用第 6 步股票池（沪深300+中证500，scripts/screener.py），
分批限速拉取并落盘 data/fundflow/<代码>.csv，支持断点续跑与状态查询。

用法示例：
    python scripts/fetch_fund_flow.py fetch --sleep 0.5   # 全池拉取（断点续跑）
    python scripts/fetch_fund_flow.py fetch --max-stocks 50
    python scripts/fetch_fund_flow.py status              # 进度/失败清单
    python scripts/fetch_fund_flow.py selftest            # 解析/落盘自检（不联网）

输出（data/fundflow/，gitignore 不进入版本库）：
  - <代码>.csv：utf-8-sig，列 date/close/pct_chg/{main,super,big,mid,small}_net
    与 *_ratio（ratio 为东财净占比，单位 %，net 单位元）；
  - fetch_state.csv：断点续跑状态；
  - fetch_failures.txt：失败清单。

免责声明：历史模拟研究仅用于学习，不构成投资建议。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 保证以 `python scripts/fetch_fund_flow.py` 方式运行时能 import 同目录模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from screener import fetch_universe  # noqa: E402

DATADIR_DEFAULT = "data/fundflow"
FETCH_STATE = "fetch_state.csv"
FETCH_FAILURES = "fetch_failures.txt"

# 东财免费接口在 http 与 https 上可用性不一致（实测 2026-11-16 本机 IP 上
# https 被限流重置、http 正常），默认 http 优先、https 回退，两个都试。
API_URLS = (
    "http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
    "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
)
API_TOKEN = "b2884a393a59ad64002292a3e90d46a5"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
}

# klines 每行逗号分隔的字段顺序（东财 daykline 接口约定，akshare 同款映射）：
# f51 日期, f52 主力净额, f53 小单净额, f54 中单净额, f55 大单净额,
# f56 超大单净额, f57 主力净占比, f58 小单净占比, f59 中单净占比,
# f60 大单净占比, f61 超大单净占比, f62 收盘价, f63 涨跌幅, f64/f65 占位
KLINE_COLS = (
    "date", "main_net", "small_net", "mid_net", "big_net", "super_net",
    "main_ratio", "small_ratio", "mid_ratio", "big_ratio", "super_ratio",
    "close", "pct_chg", "_extra1", "_extra2",
)


def parse_klines(klines: list[str]) -> pd.DataFrame:
    """把接口返回的 klines 字符串列表解析为规范 DataFrame（英文列名、数值列）。"""
    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["_extra1", "_extra2"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in list(df.columns[1:]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    return df.reset_index(drop=True)


def market_id(symbol: str) -> str:
    """东财 secid 的市场段：6 开头为沪市(1)，其余（深/北）为 0。"""
    return "1" if symbol.startswith("6") else "0"


def fetch_one(symbol: str, retries: int) -> tuple[pd.DataFrame, str]:
    """拉取单只股票的资金流历史（http 优先、https 回退），失败时指数退避重试。

    返回 (数据, 实际使用的接口 URL)。空历史（退市/长期无数据的代码）返回
    (空 DataFrame, url)，由调用方标记为 empty 状态。
    """
    import requests

    secid = f"{market_id(symbol)}.{symbol}"
    params = {
        "lmt": "0",
        "klt": "101",
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": API_TOKEN,
        "_": int(time.time() * 1000),
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for api_url in API_URLS:
            try:
                r = requests.get(api_url, params=params, headers=HEADERS, timeout=60)
                r.raise_for_status()
                data = r.json().get("data") or {}
                klines = data.get("klines") or []
                return parse_klines(klines), api_url
            except Exception as exc:  # noqa: BLE001 第三方接口，异常种类多
                last_err = exc
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"重试 {retries} 次后仍失败: {last_err}") from last_err


# ---------------------------------------------------------------------------
# 断点续跑状态
# ---------------------------------------------------------------------------

def _read_state(state_path: Path) -> pd.DataFrame:
    if not state_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(state_path, dtype={"symbol": str})
    except Exception:  # noqa: BLE001 状态文件损坏视为空
        print(f"[warn] 状态文件 {state_path} 读取失败，视为空", file=sys.stderr)
        return pd.DataFrame()


def _checkpoint_state(rows: list[dict], state_path: Path) -> None:
    """把当前拉取状态原子落盘（先写临时文件再替换），与既有状态按 symbol 合并。

    多进程并行拉取时可能同时落盘：临时文件用进程号命名，替换遇 Windows
    文件锁（WinError 32）时重试几次，避免并发 worker 崩溃。
    """
    import os

    fresh = pd.DataFrame(rows)
    if fresh.empty:
        return
    old = _read_state(state_path)
    combined = (pd.concat([old, fresh], ignore_index=True)
                .drop_duplicates("symbol", keep="last")
                if not old.empty else fresh)
    tmp_path = state_path.with_suffix(f".{os.getpid()}.tmp")
    combined.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    for attempt in range(5):
        try:
            os.replace(tmp_path, state_path)
            return
        except PermissionError:
            time.sleep(0.4 * (attempt + 1))
    print(f"[warn] 状态文件 {state_path} 落盘失败（并发锁），本次进度已随 CSV 保存",
          file=sys.stderr)


def _state_ok(state_row: pd.Series | None, csv_path: Path) -> bool:
    """状态为 ok（需 CSV 存在非空）或 empty（空历史，无需 CSV）→ 复用。"""
    if state_row is None or len(state_row) == 0:
        return False
    status = str(state_row.get("status", ""))
    if status == "empty":
        return True
    if status != "ok":
        return False
    return csv_path.exists() and csv_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> int:
    datadir = Path(args.datadir)
    datadir.mkdir(parents=True, exist_ok=True)
    state_path = datadir / FETCH_STATE
    failures_path = datadir / FETCH_FAILURES

    universe = fetch_universe(offline=False)
    symbols = universe["symbol"].tolist()
    names = dict(zip(universe["symbol"], universe["name"]))
    if args.skip:
        symbols = symbols[args.skip:]
    if args.max_stocks:
        symbols = symbols[: args.max_stocks]
    print(f"[ok] 股票池: {len(symbols)} 只（快照 {universe['as_of'].iloc[0]}，"
          f"跳过前 {args.skip} 只），资金流落盘 {datadir}")

    state = _read_state(state_path)
    state_rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    ok_count = skipped = 0
    for i, symbol in enumerate(symbols):
        csv_path = datadir / f"{symbol}.csv"
        row = None if state.empty else state[state["symbol"] == symbol]
        if _state_ok(None if row is None or row.empty else row.iloc[0], csv_path):
            skipped += 1
            continue
        if ok_count:  # 仅在两次实际请求之间限速（跳过缓存不耗时）
            time.sleep(args.sleep)
        try:
            df, api_url = fetch_one(symbol, args.retries)
            if df.empty:
                state_rows.append({
                    "symbol": symbol,
                    "status": "empty",
                    "rows": 0,
                    "first": "",
                    "last": "",
                    "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[empty] {symbol} {names.get(symbol, '')}: 无资金流历史（退市/长期停牌），"
                      f"经 {api_url.split('/')[2]}")
                continue
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            state_rows.append({
                "symbol": symbol,
                "status": "ok",
                "rows": int(len(df)),
                "first": df["date"].iloc[0].strftime("%Y-%m-%d") if len(df) else "",
                "last": df["date"].iloc[-1].strftime("%Y-%m-%d") if len(df) else "",
                "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            ok_count += 1
            print(f"[ok] {symbol} {names.get(symbol, '')}: {len(df)} 行, "
                  f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            failed.append((symbol, str(exc)))
            state_rows.append({
                "symbol": symbol,
                "status": "fail",
                "rows": 0,
                "first": "",
                "last": "",
                "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[failed] {symbol}: {exc}", file=sys.stderr, flush=True)
        if state_rows:
            _checkpoint_state(state_rows, state_path)
            state_rows = []
        if (i + 1) % 100 == 0:
            print(f"[进度] {i + 1}/{len(symbols)}（本轮成功 {ok_count}，复用 {skipped}）",
                  flush=True)

    if failed:
        with failures_path.open("w", encoding="utf-8") as fh:
            for symbol, err in failed:
                fh.write(f"{symbol}\t{err}\n")
    print(f"\n完成：成功 {ok_count}，复用 {skipped}，失败 {len(failed)}")
    if failed:
        print(f"失败清单: {failures_path}", file=sys.stderr)
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state_path = Path(args.datadir) / FETCH_STATE
    state = _read_state(state_path)
    if state.empty:
        print("[status] 暂无状态文件（尚未拉取）")
        return 0
    ok = state[state["status"] == "ok"]
    empty = state[state["status"] == "empty"]
    fail = state[state["status"] == "fail"]
    print(f"[status] 成功 {len(ok)}，空历史 {len(empty)}，失败 {len(fail)}，总计 {len(state)}")
    if len(ok):
        print(f"  覆盖范围: {ok['first'].min()} ~ {ok['last'].max()}")
    if len(fail):
        print(f"  失败: {', '.join(fail['symbol'].tolist())}")
    return 0


def cmd_selftest() -> int:
    """合成 klines 解析 + CSV 落盘/回读 + 状态合并自检（不访问网络）。"""
    import tempfile

    klines = [
        "2026-05-21,900000,-1100000,200000,300000,600000,9.0,-11.0,2.0,3.0,6.0,10.09,-0.69,-,-",
        "2026-05-22,-400000,800000,-400000,-200000,-200000,-4.0,8.0,-4.0,-2.0,-2.0,10.07,-0.20,-,-",
    ]
    df = parse_klines(klines)
    expect_cols = tuple(c for c in KLINE_COLS if not c.startswith("_extra"))
    assert list(df.columns) == list(expect_cols), f"列名异常: {df.columns}"
    assert len(df) == 2 and df["date"].iloc[0] == pd.Timestamp("2026-05-21")
    assert df["main_net"].iloc[0] == 900_000 and df["close"].iloc[1] == 10.07
    assert df["main_ratio"].iloc[1] == -4.0 and df["super_net"].iloc[1] == -200_000
    # 大单 + 超大单 = 主力 口径恒等式
    assert abs((df["big_net"] + df["super_net"] - df["main_net"]).max()) < 1e-6
    # 四个档位净流入之和 ≈ 0（买卖金额守恒）
    assert abs((df[["main_net", "mid_net", "small_net"]].sum(axis=1)).max()) < 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "000001.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        back = pd.read_csv(csv_path)
        assert len(back) == 2 and back["big_net"].iloc[0] == 300_000

        state_path = Path(tmp) / FETCH_STATE
        _checkpoint_state(
            [{"symbol": "000001", "status": "ok", "rows": 2,
              "first": "2026-05-21", "last": "2026-05-22", "at": "t1"}],
            state_path)
        _checkpoint_state(
            [{"symbol": "000002", "status": "fail", "rows": 0,
              "first": "", "last": "", "at": "t2"}],
            state_path)
        state = _read_state(state_path)
        assert len(state) == 2 and state[state["symbol"] == "000001"]["status"].iloc[0] == "ok"

    assert market_id("600519") == "1" and market_id("000001") == "0"
    print("[ok] selftest 通过：klines 解析、口径恒等式、CSV 落盘/回读、状态合并均正确")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="东财免费口径个股资金流历史拉取（issue #34）",
        epilog="示例: python scripts/fetch_fund_flow.py fetch --sleep 0.5",
    )
    parser.add_argument("--datadir", default=DATADIR_DEFAULT,
                        help=f"资金流 CSV 目录，默认 {DATADIR_DEFAULT}")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="拉取全池资金流历史（断点续跑）")
    p_fetch.add_argument("--sleep", type=float, default=0.5,
                         help="两只股票之间的请求间隔（秒），默认 0.5，勿设 0")
    p_fetch.add_argument("--retries", type=int, default=3, help="单只失败重试次数，默认 3")
    p_fetch.add_argument("--max-stocks", type=int, default=None, help="只处理前 N 只（分批用）")
    p_fetch.add_argument("--skip", type=int, default=0,
                         help="跳过前 N 只（多进程分段并行用，如 --skip 400）")

    sub.add_parser("status", help="查看拉取进度与失败清单")
    sub.add_parser("selftest", help="合成数据自检（不访问网络）")

    args = parser.parse_args(argv)
    if not args.command:
        parser.error("需要一个子命令: fetch / status / selftest")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非标准流环境无此方法
        pass
    args = parse_args(argv)
    if args.command == "selftest":
        return cmd_selftest()
    if args.command == "status":
        return cmd_status(args)
    if args.command == "fetch":
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(90)
        try:
            return cmd_fetch(args)
        finally:
            socket.setdefaulttimeout(old_timeout)
    return 2


if __name__ == "__main__":
    sys.exit(main())
