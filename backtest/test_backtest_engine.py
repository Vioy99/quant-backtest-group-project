"""AI辅助：用可手算小案例、未来数据扰动与真实附件验证内核。"""
from dataclasses import replace
from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from backtest_engine import BacktestConfig, run_backtest
from strategy_signal import generate_dual_ma_signals


def sample(signals=(1, 0, 0), opens=(10, 10, 12), closes=(10, 11, 12)):
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
        "Open": opens, "Close": closes, "Signal": signals,
        "Volume": [10000, 10000, 10000],
    })


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(initial_cash=10000)

    def test_manual_cash_shares_fees_and_next_trading_day(self):
        daily, trades = run_backtest(sample(), self.config)
        self.assertEqual(daily["Shares"].tolist(), [0, 900, 0])
        self.assertEqual(daily["Cash"].tolist(), [10000, 995, 11790])
        self.assertEqual(daily["Total_Asset"].tolist(), [10000, 10895, 11790])
        self.assertEqual(trades["Commission"].tolist(), [5, 5])
        self.assertEqual(trades["Side"].tolist(), ["BUY", "SELL"])
        # 周五信号在周一开盘成交，不是在周末，也不是信号日收盘成交。
        self.assertEqual(trades.iloc[0]["Signal_Date"], pd.Timestamp("2024-01-05"))
        self.assertEqual(trades.iloc[0]["Date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(trades.iloc[0]["Price"], 10)
        self.assertAlmostEqual(daily["Daily_Return"].iloc[1], 0.0895)
        self.assertAlmostEqual(daily["Net_Value"].iloc[-1], 1.179)

    def test_fee_aware_lot_rounding(self):
        cfg = BacktestConfig(initial_cash=100000)
        _, trades = run_backtest(sample(), cfg)
        self.assertEqual(trades.iloc[0]["Quantity"], 9900)
        self.assertAlmostEqual(trades.iloc[0]["Commission"], 29.7)
        self.assertGreaterEqual(trades.iloc[0]["Cash_After"], 0)

    def test_persistent_one_is_target_not_repeated_order(self):
        daily, trades = run_backtest(sample(signals=(1, 1, 1)), self.config)
        self.assertEqual(len(trades), 1)
        self.assertEqual(daily["Shares"].iloc[-1], 900)
        self.assertEqual(daily["Total_Asset"].iloc[-1], 11795)

    def test_final_signal_not_filled_and_one_row_valid(self):
        daily, trades = run_backtest(sample(signals=(0, 0, 1)), self.config)
        self.assertTrue(trades.empty)
        self.assertEqual(daily["Net_Value"].tolist(), [1, 1, 1])
        daily, trades = run_backtest(sample().iloc[:1], self.config)
        self.assertTrue(trades.empty)
        self.assertEqual(daily["Shares"].iloc[0], 0)

    def test_insufficient_cash_never_borrows(self):
        daily, trades = run_backtest(sample(), replace(self.config, initial_cash=100))
        self.assertTrue(trades.empty)
        self.assertEqual(daily["Order_Status"].iloc[1], "INSUFFICIENT_CASH")
        self.assertEqual(daily["Cash"].tolist(), [100, 100, 100])

    def test_slippage_and_sell_tax_not_double_deducted(self):
        cfg = replace(self.config, slippage_rate=0.01, sell_tax_rate=0.001)
        daily, trades = run_backtest(sample(), cfg)
        self.assertAlmostEqual(trades.iloc[0]["Price"], 10.1)
        self.assertAlmostEqual(trades.iloc[1]["Price"], 11.88)
        self.assertAlmostEqual(trades.iloc[1]["Stamp_Duty"], 10.692)
        self.assertAlmostEqual(daily["Total_Asset"].iloc[-1], 11581.308)
        self.assertAlmostEqual(trades["Slippage_Cost"].sum(), 198)

    def test_zero_volume_blocks_then_uses_latest_target(self):
        frame = sample(signals=(1, 1, 0))
        frame.loc[1, "Volume"] = 0
        daily, trades = run_backtest(frame, self.config)
        self.assertEqual(daily["Order_Status"].iloc[1], "BLOCKED_ZERO_VOLUME")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Signal_Date"], frame.loc[1, "Date"])
        self.assertEqual(trades.iloc[0]["Date"], frame.loc[2, "Date"])

    def test_bad_inputs_rejected(self):
        cases = []
        duplicate = sample(); duplicate.loc[1, "Date"] = duplicate.loc[0, "Date"]; cases.append(duplicate)
        cases.append(sample().iloc[::-1])
        for column, value in [("Open", 0), ("Close", float("nan")), ("Close", float("inf")),
                              ("Signal", -1), ("Signal", 0.5), ("Volume", -1)]:
            frame = sample(); frame[column] = frame[column].astype(float)
            frame.loc[1, column] = value; cases.append(frame)
        cases.extend([sample().drop(columns="Open"), sample().iloc[:0]])
        for frame in cases:
            with self.subTest(frame=frame.to_string()):
                with self.assertRaises(ValueError): run_backtest(frame, self.config)

    def test_bad_config_rejected(self):
        for kwargs in ({"initial_cash": 0}, {"commission_rate": -1}, {"slippage_rate": 1},
                       {"min_commission": float("nan")}, {"lot_size": 1.5},
                       {"lot_size": True}, {"sell_tax_rate": 0.8, "commission_rate": 0.3}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): BacktestConfig(**kwargs)

    def test_date_column_and_index_equivalent_input_unchanged(self):
        frame = sample(); before = frame.copy(deep=True)
        daily, trades = run_backtest(frame, self.config)
        indexed_daily, indexed_trades = run_backtest(frame.set_index("Date"), self.config)
        assert_frame_equal(daily, indexed_daily)
        assert_frame_equal(trades, indexed_trades)
        assert_frame_equal(frame, before)

    def test_future_changes_do_not_change_past(self):
        root = Path(__file__).resolve().parent
        frame = pd.read_csv(root / "data/maotai_test_signals.csv")
        prefix_daily, prefix_trades = run_backtest(frame.iloc[:100], self.config)
        full_daily, full_trades = run_backtest(frame, self.config)
        assert_frame_equal(prefix_daily, full_daily.iloc[:100])
        assert_frame_equal(prefix_trades, full_trades[full_trades["Date"] <= prefix_daily.index[-1]])
        altered = frame.copy()
        altered.loc[100:, ["Open", "Close"]] *= 2
        altered.loc[100:, "Signal"] = 1 - altered.loc[100:, "Signal"]
        new_daily, new_trades = run_backtest(altered, self.config)
        assert_frame_equal(prefix_daily, new_daily.iloc[:100])
        assert_frame_equal(prefix_trades, new_trades[new_trades["Date"] <= prefix_daily.index[-1]])

    def test_real_data_strategy_is_causal_and_account_reconciles(self):
        root = Path(__file__).resolve().parent
        raw = pd.read_csv(root / "data/maotai_total_data.csv", parse_dates=["Date"]).set_index("Date")
        full = generate_dual_ma_signals(raw)
        prefix = generate_dual_ma_signals(raw.iloc[:800])
        assert_frame_equal(prefix, full.loc[prefix.index])
        test = full.loc["2023-01-01":"2024-12-31"]
        daily, trades = run_backtest(test)
        self.assertEqual(len(daily), 484)
        self.assertTrue((daily["Cash"] >= 0).all())
        self.assertTrue((daily["Shares"] % 100 == 0).all())
        self.assertLess((daily["Total_Asset"] - daily["Cash"] - daily["Shares"] * daily["Close"]).abs().max(), 1e-8)
        cash, shares = 1000000, 0
        for _, trade in trades.iterrows():
            if trade["Side"] == "BUY":
                cash -= trade["Trade_Value"] + trade["Commission"]
                shares += trade["Quantity"]
            else:
                cash += trade["Trade_Value"] - trade["Commission"] - trade["Stamp_Duty"]
                shares -= trade["Quantity"]
            self.assertAlmostEqual(cash, trade["Cash_After"])
            self.assertEqual(shares, trade["Shares_After"])
            self.assertEqual(daily.index.get_loc(trade["Date"]), daily.index.get_loc(trade["Signal_Date"]) + 1)
        self.assertAlmostEqual(cash, daily["Cash"].iloc[-1])
        self.assertEqual(shares, daily["Shares"].iloc[-1])
        self.assertAlmostEqual(trades["Commission"].sum(), daily["Commission"].sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
