import akshare as ak
import pandas as pd

print("正在下载沪深300基准数据...")
# 获取沪深300指数
df = ak.stock_zh_index_daily(symbol="sh000300")

# 提取收盘价并清洗
df = df[['date', 'close']]
df.columns = ['Date', 'Close']
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# 截取 2020-2024 年的数据
df = df.loc['2020-01-01':'2024-12-31']

# 保存
df.to_csv('benchmark_hs300.csv')
print("沪深300数据已保存为 benchmark_hs300.csv，搞定！")