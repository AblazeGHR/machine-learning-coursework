import pandas as pd
from sklearn.datasets import fetch_california_housing

# 方式1：直接加载数据集（自动下载/读取本地pkz文件）
housing = fetch_california_housing()

# 1. 查看房价（target）是否存在
print("✅ 房价数据是否存在：", hasattr(housing, "target"))
print("✅ 房价数据前5个值：", housing.target[:5])  # 输出：[4.526 3.585 3.521 3.413 3.422]
print("✅ 房价字段名：", housing.target_names)  # 输出：['MedHouseVal']（房屋中位数价值）

# 2. 合并特征+房价为完整DataFrame（这才是包含房价的完整数据）
df = pd.DataFrame(
    data=housing.data,       # 特征列
    columns=housing.feature_names  # 特征名
)
df[housing.target_names[0]] = housing.target  # 把房价作为新列加入

# 3. 查看最终结果（包含房价）
print("\n📊 完整数据（含房价）前5行：")
print(df.head())