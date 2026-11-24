"""组员3联调入口（AI辅助）：使用原版组员2策略，固定5/20，不做参数寻优。

历史行情仅用于均线预热；样本内外各自以同额现金独立启动。
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_engine import BacktestConfig, run_backtest, save_results
from strategy_signal import generate_dual_ma_signals


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    args = parser.parse_args()
    config = BacktestConfig(initial_cash=args.initial_cash)
    source = args.data_dir / "maotai_total_data.csv"
    raw = pd.read_csv(source, parse_dates=["Date"]).set_index("Date")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("行情必须按日期升序，且不重复")
    if not raw.index.equals(raw.index.normalize()):
        raise ValueError("需要日频行情")

    # 组员2函数保持原样：rolling仅使用当前日及过去数据。
    generated = generate_dual_ma_signals(raw, short_window=5, long_window=20)
    if not generated.index.equals(raw.index[19:]):
        raise ValueError("策略输出有预热期以外的缺失，请检查上游数据")
    signals = raw.copy()
    for name in ("MA_Short", "MA_Long", "Signal", "Position_Change"):
        signals[name] = generated[name].reindex(raw.index)
    # 全历史最初19日均线不足时等待；保留这些现金日，统一回测和基准起点。
    signals.loc[raw.index[:19], "Signal"] = 0
    signals["Signal"] = signals["Signal"].astype(int)
    signals["Position_Change"] = signals["Signal"].diff().fillna(0)

    benchmark_path = args.data_dir / "benchmark_hs300.csv"
    benchmark = pd.read_csv(benchmark_path, parse_dates=["Date"]).set_index("Date")
    if not benchmark.index.is_unique or not benchmark.index.is_monotonic_increasing:
        raise ValueError("基准日期必须升序且不重复")
    cases = {
        "train": signals.loc["2020-01-01":"2022-12-31"],
        "test": signals.loc["2023-01-01":"2024-12-31"],
        # 单独复现组员2已交付的465行文件，不与完整样本外口径混用。
        "provided_signals": pd.read_csv(args.data_dir / "maotai_test_signals.csv"),
    }
    summary = {}
    for label, frame in cases.items():
        daily, trades = run_backtest(frame, config)
        aligned = benchmark.reindex(daily.index)["Close"]
        if aligned.isna().any() or not (aligned > 0).all():
            raise ValueError("基准存在缺失/无效价格；不自动填充或删交易日")
        daily["Benchmark_Close"] = aligned
        daily["Benchmark_Net_Value"] = aligned / aligned.iloc[0]
        dest = args.output_dir / label
        save_results(daily, trades, dest)
        frame.to_csv(dest / "input_signals.csv", index=label != "provided_signals", encoding="utf-8-sig")
        summary[label] = {
            "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
            "trading_days": len(daily), "trades": len(trades),
            "buy_trades": int((trades["Side"] == "BUY").sum()),
            "sell_trades": int((trades["Side"] == "SELL").sum()),
            "final_cash": float(daily["Cash"].iloc[-1]),
            "final_shares": int(daily["Shares"].iloc[-1]),
            "final_total_asset": float(daily["Total_Asset"].iloc[-1]),
            "final_net_value": float(daily["Net_Value"].iloc[-1]),
            "total_commission": float(trades["Commission"].sum()),
        }
        print(label, json.dumps(summary[label], ensure_ascii=False))
    metadata = {
        "config": asdict(config), "short_window": 5, "long_window": 20,
        "parameter_search_performed": False,
        "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (source, benchmark_path, args.data_dir / "maotai_test_signals.csv")},
        "results": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
