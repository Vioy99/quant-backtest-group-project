import pandas as pd

print("开始执行双均线策略信号生成...")

def generate_dual_ma_signals(df, short_window=5, long_window=20):
    """
    双均线策略核心逻辑
    :param df: 包含 Open, Close, High, Low, Volume 的 DataFrame
    :param short_window: 短周期均线（默认5天）
    :param long_window: 长周期均线（默认20天）
    :return: 添加了均线、信号、买卖点标记的 DataFrame
    """
    df = df.copy()
    
    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_window).mean()
    
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    df.loc[df['MA_Short'] <= df['MA_Long'], 'Signal'] = 0
    
    df['Position_Change'] = df['Signal'].diff()
    
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    print("\n正在读取数据...")
    try:
        df = pd.read_csv('maotai_test_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    except FileNotFoundError:
        print("找不到 CSV 文件，请确认组员1的数据是否在当前目录下！")
        exit()

    df_with_signals = generate_dual_ma_signals(df, short_window=5, long_window=20)

    print("\n--- 信号生成完成！部分数据长这样：---")
    print(df_with_signals[['Close', 'MA_Short', 'MA_Long', 'Signal', 'Position_Change']].head())
    
    buy_count = len(df_with_signals[df_with_signals['Position_Change'] == 1])
    sell_count = len(df_with_signals[df_with_signals['Position_Change'] == -1])
    print(f"\n--- 策略统计 ---")
    print(f"样本期内共产生 {buy_count} 次买入信号，{sell_count} 次卖出信号。")

    df_with_signals.to_csv('maotai_test_signals.csv')
    print("\n带有信号的完整数据已保存为 maotai_test_signals.csv")