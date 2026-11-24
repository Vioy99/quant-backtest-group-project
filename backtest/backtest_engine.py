"""组员3：手写单股票、只做多的日频回测内核。

AI 辅助：本文件由 AI 辅助编写，须由组员阅读、复核并在报告披露。
Signal 是目标状态（1持仓、0空仓），不是逐日买卖指令。
收盘信号在下一条交易日记录的 Open 执行；收盘 Close 估值。
仅依赖 pandas，不调用第三方回测引擎。Python >= 3.10。
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    lot_size: int = 100
    slippage_rate: float = 0.0
    sell_tax_rate: float = 0.0  # 教学简化；如设置，为整个样本的固定卖出税率

    def __post_init__(self):
        for name in ("initial_cash", "commission_rate", "min_commission",
                     "slippage_rate", "sell_tax_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是有限数值")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash 必须大于0")
        if isinstance(self.lot_size, bool) or not isinstance(self.lot_size, Integral) or self.lot_size <= 0:
            raise ValueError("lot_size 必须是正整数")
        if any(getattr(self, x) >= 1 for x in ("commission_rate", "slippage_rate", "sell_tax_rate")):
            raise ValueError("费率及滑点比例必须小于1")
        if self.commission_rate + self.sell_tax_rate >= 1:
            raise ValueError("佣金与卖出税率之和必须小于1")


TRADE_COLUMNS = [
    "Signal_Date", "Date", "Side", "Open_Price", "Price", "Quantity",
    "Trade_Value", "Commission", "Stamp_Duty", "Slippage_Cost",
    "Cash_After", "Shares_After",
]


def _prepare_data(data):
    """不偷偷排序或删除行：信号日期错误时直接报错，交还上游修复。"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data 必须是非空 DataFrame")
    df = data.copy(deep=True)
    if not df.columns.is_unique:
        raise ValueError("列名不能重复")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("请提供 Date 列或 DatetimeIndex 日期索引")
    df.index.name = "Date"
    if df.index.hasnans or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("日期必须有效、不重复，并按升序排列")
    if not df.index.equals(df.index.normalize()):
        raise ValueError("本内核要求日频日期（不包含日内时间）")
    required = {"Open", "Close", "Signal"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必需列：{sorted(required - set(df.columns))}")
    for name in ("Open", "Close", "Signal", "Volume"):
        if name not in df.columns:
            continue
        df[name] = pd.to_numeric(df[name], errors="raise")
        if df[name].isna().any() or not df[name].map(math.isfinite).all():
            raise ValueError(f"{name} 存在缺失或非有限数值")
    if (df[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Open 和 Close 必须大于0")
    if not df["Signal"].isin([0, 1]).all():
        raise ValueError("Signal 必须为目标状态0或1；不要传入Position_Change")
    if "Volume" in df and (df["Volume"] < 0).any():
        raise ValueError("Volume 不能为负数")
    return df


def run_backtest(data, config=None):
    """返回 (daily, trades)。daily 为日期索引；trades 每行是一笔实际成交。

    首日空仓且不交易；第i日只读取第i-1日的Signal作为目标状态。
    买入时用可用现金买最多整手；不加仓、不借款、不做空。
    卖出时卖出全部持仓；末日持仓按收盘估值，不强制平仓。
    前提：上游 Signal 已因果计算，内核不能识别上游偷看未来的信号。
    """
    config = config or BacktestConfig()
    df = _prepare_data(data)
    cash = float(config.initial_cash)
    shares = 0
    previous_asset = cash
    account_rows, trades = [], []

    # 真实回测内核：逐日执行前一交易日信号，再以当天收盘价估值。
    for i, (date, row) in enumerate(df.iterrows()):
        target = None if i == 0 else int(df.iloc[i - 1]["Signal"])
        signal_date = pd.NaT if i == 0 else df.index[i - 1]
        commission = stamp_duty = slippage_cost = 0.0
        buy_marker = sell_marker = float("nan")
        status = "NO_PREVIOUS_SIGNAL" if i == 0 else "HOLD"
        side = "BUY" if target == 1 and shares == 0 else (
            "SELL" if target == 0 and shares > 0 else None
        )

        # Volume 为0时不模拟成交；下一日按最新已知目标状态再次判断。
        if side and "Volume" in df.columns and row["Volume"] == 0:
            status = "BLOCKED_ZERO_VOLUME"
        elif side:
            open_price = float(row["Open"])
            price = open_price * (1 + config.slippage_rate if side == "BUY"
                                  else 1 - config.slippage_rate)
            if side == "BUY":
                # 同时考虑比例佣金与最低佣金，确保买入后现金不为负。
                affordable_value = min(cash / (1 + config.commission_rate),
                                       cash - config.min_commission)
                lots = max(0, math.floor(affordable_value / (price * config.lot_size)))
                quantity = lots * config.lot_size
            else:
                quantity = shares

            value = quantity * price
            fee = max(value * config.commission_rate, config.min_commission) if quantity else 0.0
            tax = value * config.sell_tax_rate if side == "SELL" else 0.0
            # 浮点边界检查；整手计算后仍不可用超额现金买入。
            if side == "BUY" and quantity and value + fee > cash:
                quantity -= config.lot_size
                value = quantity * price
                fee = max(value * config.commission_rate, config.min_commission) if quantity else 0.0
            if quantity == 0:
                status = "INSUFFICIENT_CASH"
            elif side == "SELL" and cash + value - fee - tax < 0:
                status = "INSUFFICIENT_CASH_FOR_SELL_FEES"
            else:
                if side == "BUY":
                    cash -= value + fee
                    shares += quantity
                    buy_marker = price
                else:
                    cash += value - fee - tax
                    shares -= quantity
                    sell_marker = price
                commission, stamp_duty = fee, tax
                slippage_cost = abs(price - open_price) * quantity
                status = "FILLED_" + side
                trades.append({
                    "Signal_Date": signal_date, "Date": date, "Side": side,
                    "Open_Price": open_price, "Price": price, "Quantity": quantity,
                    "Trade_Value": value, "Commission": fee, "Stamp_Duty": tax,
                    "Slippage_Cost": slippage_cost, "Cash_After": cash,
                    "Shares_After": shares,
                })

        position_value = shares * float(row["Close"])
        total_asset = cash + position_value
        account_rows.append({
            "Execution_Target": target, "Execution_Signal_Date": signal_date,
            "Cash": cash, "Shares": shares, "Position_Value": position_value,
            "Total_Asset": total_asset, "Net_Value": total_asset / config.initial_cash,
            "Daily_Return": total_asset / previous_asset - 1,
            "Commission": commission, "Stamp_Duty": stamp_duty,
            "Slippage_Cost": slippage_cost, "Buy_Marker": buy_marker,
            "Sell_Marker": sell_marker, "Order_Status": status,
        })
        previous_asset = total_asset

    # 保留上游行情、均线和信号，方便组员4按成交日期绘制买卖点。
    daily = df.copy()
    for name in account_rows[0]:
        daily[name] = [r[name] for r in account_rows]
    return daily, pd.DataFrame(trades, columns=TRADE_COLUMNS)


def save_results(daily, trades, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_account.csv", encoding="utf-8-sig")
    trades.to_csv(output_dir / "trade_records.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="带Date、Open、Close、Signal列的CSV路径")
    parser.add_argument("--output-dir", default="results/direct")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--slippage-rate", type=float, default=0)
    parser.add_argument("--sell-tax-rate", type=float, default=0)
    args = parser.parse_args()
    config = BacktestConfig(**{k: getattr(args, k) for k in asdict(BacktestConfig())})
    daily, trades = run_backtest(pd.read_csv(args.input), config)
    save_results(daily, trades, args.output_dir)
    (Path(args.output_dir) / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"交易日 {len(daily)}，成交 {len(trades)} 笔，期末总资产 {daily['Total_Asset'].iloc[-1]:,.2f}")
    print(f"输出目录：{Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
