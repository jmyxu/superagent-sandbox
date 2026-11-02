"""用 akshare 拉取 A 股日线数据（默认前复权）并保存为 CSV。

用法示例：
    # 拉取平安银行和贵州茅台 2023 年全年日线（前复权）
    python scripts/fetch_daily.py 000001 600519 --start 20230101 --end 20231231

    # 拉取全部历史，后复权，输出到指定目录
    python scripts/fetch_daily.py 000001 --adjust hfq --outdir data/daily

说明：
- 股票代码为 6 位数字（如 000001 / 600519），也兼容 "sh600519" 这类带前缀的写法；
- 复权方式：qfq 前复权（默认）/ hfq 后复权 / none 不复权；
- 输出文件名形如 <代码>_<复权方式>.csv，编码 utf-8-sig，Excel 可直接打开；
- 重复运行会覆盖同名 CSV；如需增量更新，请自行与历史数据按 (代码, 日期) 去重合并。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# akshare 上游把 19700101 视为"不限制起始日期"
HISTORY_START = "19700101"


def normalize_symbol(raw: str) -> str:
    """把 'sh600519' / '600519.SH' 之类的写法归一成 6 位数字代码。"""
    match = re.search(r"\d{6}", raw.strip())
    if not match:
        raise ValueError(f"无法识别股票代码: {raw!r}（应为 6 位数字，如 000001）")
    return match.group(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 akshare 拉取 A 股日线数据（含前复权）并保存 CSV",
        epilog="示例: python scripts/fetch_daily.py 000001 600519 --start 20230101 --end 20231231",
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="股票代码，可传多个，如 000001 600519（也兼容 sh600519 写法）",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="开始日期 YYYYMMDD，默认不限（取全部历史）",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="结束日期 YYYYMMDD，默认今天",
    )
    parser.add_argument(
        "--adjust",
        choices=["qfq", "hfq", "none"],
        default="qfq",
        help="复权方式: qfq 前复权(默认) / hfq 后复权 / none 不复权",
    )
    parser.add_argument(
        "--outdir",
        default="data/daily",
        help="CSV 输出目录，默认 data/daily",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="单只股票失败重试次数，默认 3",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="两只股票之间的请求间隔（秒），默认 1，请勿设 0 以免对上游造成压力",
    )
    return parser.parse_args(argv)


def fetch_one(
    symbol: str,
    start: str | None,
    end: str | None,
    adjust: str,
    retries: int,
) -> pd.DataFrame:
    """拉取单只股票的日线数据，失败时按指数退避重试。"""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 akshare，请先执行: pip install akshare pandas") from exc

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    adjust_param = "" if adjust == "none" else adjust

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start or HISTORY_START,
                end_date=end,
                adjust=adjust_param,
            )
            if df is None or df.empty:
                raise ValueError("接口返回空数据（代码可能有误，或该区间内无交易）")
            return df
        except Exception as exc:  # noqa: BLE001 上游为第三方接口，异常种类多，统一重试
            last_err = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"重试 {retries} 次后仍失败: {last_err}") from last_err


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for i, raw_symbol in enumerate(args.symbols):
        if i > 0:
            time.sleep(args.sleep)
        try:
            symbol = normalize_symbol(raw_symbol)
        except ValueError as exc:
            print(f"[failed] {raw_symbol}: {exc}", file=sys.stderr)
            failed.append(raw_symbol)
            continue
        try:
            df = fetch_one(symbol, args.start, args.end, args.adjust, args.retries)
            out_path = outdir / f"{symbol}_{args.adjust}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(
                f"[ok] {symbol}: {len(df)} 行, "
                f"{df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}, 已保存 {out_path}"
            )
        except Exception as exc:  # noqa: BLE001 单只失败不中断批量任务
            print(f"[failed] {symbol}: {exc}", file=sys.stderr)
            failed.append(symbol)

    if failed:
        print(f"共 {len(args.symbols)} 只，失败 {len(failed)} 只: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"全部成功：{len(args.symbols)} 只")
    return 0


if __name__ == "__main__":
    sys.exit(main())
