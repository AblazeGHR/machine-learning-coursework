import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. 加载 California Housing 数据集
data = fetch_california_housing()
X = data.data
y = data.target

# 进行 80% 训练集 / 20% 测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基准模型：使用训练集目标变量的平均值作为预测
y_pred_baseline = np.mean(y_train) * np.ones(len(y_test))
r2_baseline = r2_score(y_test, y_pred_baseline)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print("基准模型（训练集平均值预测）结果:")
print(f"R² 分数: {r2_baseline:.4f}")
print(f"MSE: {mse_baseline:.4f}")
print(f"RMSE: {rmse_baseline:.4f}")
print(f"MAE: {mae_baseline:.4f}")

# 2. 对特征做标准化预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 使用 5 折交叉验证搜索 ElasticNet 的最优超参数
param_grid1 = {
    'alpha': [0.001,0.01, 0.1, 1, 10, 100,1000],
    'l1_ratio': [0.1, 0.3,0.5,0.7, 0.9]
}
param_grid2 = {
    'alpha': [0.01, 0.1, 1, 10, 100,1000],
    'l1_ratio': [0.1, 0.3,0.5,0.7, 0.9]
}

grid_search = GridSearchCV(ElasticNet(), param_grid2, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print(f"\nElastic-Net 最优超参数: {best_params}")

# 4. 用最优超参数在完整训练集上训练模型
model = ElasticNet(**best_params)
model.fit(X_train_scaled, y_train)

# 在测试集上评估预测性能
y_pred_en = model.predict(X_test_scaled)
r2_en = r2_score(y_test, y_pred_en)
mse_en = mean_squared_error(y_test, y_pred_en)
rmse_en = np.sqrt(mse_en)
mae_en = mean_absolute_error(y_test, y_pred_en)

print("Elastic-Net 模型结果:")
print(f"R² 分数: {r2_en:.4f}")
print(f"MSE: {mse_en:.4f}")
print(f"RMSE: {rmse_en:.4f}")
print(f"MAE: {mae_en:.4f}")

# 5. 模型对比
print("\n模型对比分析:")
print(f"- 基准模型 R²: {r2_baseline:.4f}，Elastic-Net R²: {r2_en:.4f} (提升: {r2_en - r2_baseline:.4f})")
print(f"- 基准模型 MSE: {mse_baseline:.4f}，Elastic-Net MSE: {mse_en:.4f} (降低: {mse_baseline - mse_en:.4f})")
print(f"- 基准模型 RMSE: {rmse_baseline:.4f}，Elastic-Net RMSE: {rmse_en:.4f} (降低: {rmse_baseline - rmse_en:.4f})")
print(f"- 基准模型 MAE: {mae_baseline:.4f}，Elastic-Net MAE: {mae_en:.4f} (降低: {mae_baseline - mae_en:.4f})")
print("- Elastic-Net 显著优于基准模型，表明机器学习模型能有效捕捉房价预测的模式，而非简单平均。")
