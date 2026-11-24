# %%

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 全局绘图配置（解决中文、清晰度） --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG_DPI = 300  # 导出保存的图片分辨率


def load_data():
    """读取上游输出csv，日期列转为datetime，设置索引
    输入文件由其他模块生成，本模块只负责读取绘图
    maotai_test_signals.csv
    backtest_account_result.csv
    backtest_performance.csv
    hs300_daily.csv
    """
    try:
        df_signal = pd.read_csv("maotai_test_signals.csv", encoding="utf-8-sig", parse_dates=["Date"])
        df_signal = df_signal.set_index("Date")

        df_account = pd.read_csv("backtest_account_result.csv", encoding="utf-8-sig", parse_dates=["Date"])
        df_account = df_account.set_index("Date")

        df_perf = pd.read_csv("backtest_performance.csv", encoding="utf-8-sig")

        df_benchmark = pd.read_csv("hs300_daily.csv", encoding="utf-8-sig", parse_dates=["Date"])
        df_benchmark = df_benchmark.set_index("Date")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"缺少上游输出CSV文件，请确认csv文件和notebook在同一个目录下: {e}")

    return df_signal, df_account, df_perf, df_benchmark


def plot_price_signal(df_signal):
    """
    图1：行情与均线信号图 price_signal_plot.png
    绘制收盘价、短期/长期均线；标记金叉买入▲、死叉卖出▼
    底色区分样本内(2020‑2022)、样本外(2023‑2024)
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df_signal.index, df_signal["Close"], label="前复权收盘价", color="#2c3e50", lw=1.2)
    ax.plot(df_signal.index, df_signal["MA_Short"], label="短期均线MA_Short", color="#e74c3c", lw=1.5)
    ax.plot(df_signal.index, df_signal["MA_Long"], label="长期均线MA_Long", color="#3498db", lw=1.5)

    buy_points = df_signal[df_signal["Position_Change"] == 1]
    sell_points = df_signal[df_signal["Position_Change"] == -1]

    ax.scatter(buy_points.index, buy_points["Close"], marker="^", color="#c0392b", s=80, zorder=5, label="金叉买入")
    ax.scatter(sell_points.index, sell_points["Close"], marker="v", color="#27ae60", s=80, zorder=5, label="死叉卖出")

    sample_split = pd.to_datetime("2022-12-31")
    ax.axvspan(df_signal.index.min(), sample_split, alpha=0.08, color="#95a5a6", label="样本内(训练集2020‑2022)")
    ax.axvspan(sample_split, df_signal.index.max(), alpha=0.08, color="#f39c12", label="样本外(验证集2023‑2024)")

    ax.set_title("贵州茅台：双均线策略行情与交易信号", fontsize=14, pad=15)
    ax.set_ylabel("前复权价格", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("price_signal_plot.png", dpi=FIG_DPI, bbox_inches="tight")
    # jupyter环境不调用plt.close()，图片才能在cell显示
    print("✅已输出文件：price_signal_plot.png")


def plot_net_value_compare(df_account, df_benchmark):
    """
    图2：净值对比曲线图 net_value_compare.png
    策略累计净值 vs 沪深300基准净值
    """
    fig, ax = plt.subplots(figsize=(16,7))

    strategy_net = df_account["Total_Asset"] / df_account["Total_Asset"].iloc[0]
    benchmark_net = df_benchmark["Close"] / df_benchmark["Close"].iloc[0]

    ax.plot(strategy_net.index, strategy_net.values, color="#d35400", lw=1.6, label="双均线策略净值")
    ax.plot(benchmark_net.index, benchmark_net.values, color="#2980b9", lw=1.4, label="沪深300基准净值")

    sample_split = pd.to_datetime("2022-12-31")
    ax.axvspan(strategy_net.index.min(), sample_split, alpha=0.08, color="#95a5a6", label="样本内(训练集2020‑2022)")
    ax.axvspan(sample_split, strategy_net.index.max(), alpha=0.08, color="#f39c12", label="样本外(验证集2023‑2024)")

    ax.set_title("策略净值 vs 沪深300基准净值对比", fontsize=14, pad=15)
    ax.set_ylabel("归一化净值(初始=1)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("net_value_compare.png", dpi=FIG_DPI, bbox_inches="tight")
    print("✅已输出文件：net_value_compare.png")


def plot_drawdown(df_account):
    """
    图3：回撤时序图 drawdown_plot.png
    根据账户总资产计算滚动最大回撤时序曲线
    """
    fig, ax = plt.subplots(figsize=(16,6))
    total_asset = df_account["Total_Asset"]
    rolling_max = total_asset.cummax()
    drawdown = (total_asset - rolling_max) / rolling_max

    ax.fill_between(drawdown.index, drawdown.values, y2=0, color="#e74c3c", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="#c0392b", lw=1)

    ax.set_title("策略历史回撤时序", fontsize=14, pad=15)
    ax.set_ylabel("回撤", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("drawdown_plot.png", dpi=FIG_DPI, bbox_inches="tight")
    print("✅已输出文件：drawdown_plot.png")


def plot_monthly_return_bar(df_account):
    """
    图4：月度收益柱状图 monthly_return_bar.png
    由账户每日总资产计算月度收益率
    """
    fig, ax = plt.subplots(figsize=(18,6))
    total_asset = df_account["Total_Asset"]
    monthly_asset = total_asset.resample("M").last()
    monthly_ret = monthly_asset.pct_change()

    colors = np.where(monthly_ret >= 0, "#27ae60", "#e74c3c")
    ax.bar(monthly_ret.index, monthly_ret.values, color=colors, width=20)

    ax.set_title("策略月度收益率", fontsize=14, pad=15)
    ax.set_ylabel("月度收益率", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("monthly_return_bar.png", dpi=FIG_DPI, bbox_inches="tight")
    print("✅已输出文件：monthly_return_bar.png")


def main():
    """可视化模块主入口，只做读取+绘图，无任何策略、回测逻辑"""
    print("===== 可视化绘图模块开始执行(Jupyter环境) =====")
    df_signal, df_account, df_perf, df_benchmark = load_data()

    plot_price_signal(df_signal)
    plot_net_value_compare(df_account, df_benchmark)
    plot_drawdown(df_account)
    plot_monthly_return_bar(df_account)

    print("\n===== ✨全部图表输出完成，png保存在当前工作目录 =====")


# %%
if __name__ == "__main__":
    main()
