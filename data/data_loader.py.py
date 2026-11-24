import akshare as ak
import pandas as pd

print("正在从新浪接口下载 贵州茅台 的数据...")

# 1. 获取数据 (新浪接口，带 'sh' 前缀)
df = ak.stock_zh_a_daily(symbol="sh600519", start_date="20200101", end_date="20241231", adjust="qfq")

# ================= 2. 数据清洗 =================
# 提取需要的列
df = df[['date', 'open', 'close', 'high', 'low', 'volume']]

# 把列名改成首字母大写，方便队友使用
df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'])

# 重点修改：去掉了 inplace=True，直接用 df = df.xxx() 的方式
df = df.set_index('Date')
df = df.sort_index()
df = df.dropna()

print("\n--- 数据清洗完成！前5行长这样：---")
print(df.head())

# ================= 3. 样本内/外切分 =================
train_data = df.loc['2020-01-01':'2022-12-31']
test_data = df.loc['2023-01-01':'2024-12-31']

print("\n--- 切分完成！---")
print(f"样本内数据（2020-2022）共有 {len(train_data)} 个交易日")
print(f"样本外数据（2023-2024）共有 {len(test_data)} 个交易日")

# ================= 4. 保存文件，交接给队友 =================
df.to_csv('maotai_total_data.csv')
train_data.to_csv('maotai_train_data.csv')
test_data.to_csv('maotai_test_data.csv')

print("\n数据已经保存为 CSV 文件！任务完成！")