import numpy as np
from sklearn.datasets import fetch_california_housing

# 加载加州住房数据集
data = fetch_california_housing()
y = data.target  # 房价数据

# 打印数据集大小
print(f"数据集样本数: {y.shape[0]}")

# 计算平均房价
mean_price = np.mean(y)

# 计算中位数房价
median_price = np.median(y)

# 输出结果
print(f"加利福尼亚房价平均值: {mean_price:.4f}")
print(f"加利福尼亚房价中位数: {median_price:.4f}")
