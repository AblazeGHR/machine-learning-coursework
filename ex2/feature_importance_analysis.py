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
from sklearn.metrics import r2_score, mean_squared_error

# 1. 加载 California Housing 数据集
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# 进行 80% 训练集 / 20% 测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print(f"Elastic-Net 最优超参数: {best_params}")

# 4. 用最优超参数在完整训练集上训练模型
model_full = ElasticNet(**best_params)
model_full.fit(X_train_scaled, y_train)

# 在测试集上评估完整特征模型的预测性能
y_pred_full = model_full.predict(X_test_scaled)
r2_full = r2_score(y_test, y_pred_full)
mse_full = mean_squared_error(y_test, y_pred_full)

print("完整特征 Elastic-Net 模型结果:")
print(f"R² 分数: {r2_full:.4f}")
print(f"MSE: {mse_full:.4f}")

# 5. 提取特征重要性（系数绝对值）
feature_importance = np.abs(model_full.coef_)
importance_indices = np.argsort(feature_importance)[::-1]  # 降序排序索引

print("\n特征重要性排序（从高到低）:")
for i, idx in enumerate(importance_indices):
    print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# 找出最重要的3个特征
top3_indices = importance_indices[:3]
top3_features = [feature_names[idx] for idx in top3_indices]
print(f"\n最重要的3个特征: {top3_features}")

# 6. 仅使用这3个最重要特征，重新训练 ElasticNet 模型
X_train_top3 = X_train[:, top3_indices]
X_test_top3 = X_test[:, top3_indices]

# 重新标准化（基于这3个特征）
scaler_top3 = StandardScaler()
X_train_top3_scaled = scaler_top3.fit_transform(X_train_top3)
X_test_top3_scaled = scaler_top3.transform(X_test_top3)

# 使用相同超参数训练新模型
model_top3 = ElasticNet(**best_params)
model_top3.fit(X_train_top3_scaled, y_train)

# 在测试集上评估新模型
y_pred_top3 = model_top3.predict(X_test_top3_scaled)
r2_top3 = r2_score(y_test, y_pred_top3)
mse_top3 = mean_squared_error(y_test, y_pred_top3)

print("\n仅使用前3个特征的 Elastic-Net 模型结果:")
print(f"R² 分数: {r2_top3:.4f}")
print(f"MSE: {mse_top3:.4f}")

# 7. 性能对比
r2_drop = r2_full - r2_top3
mse_increase = mse_top3 - mse_full

print("\n性能对比:")
print(f"- R² 分数下降: {r2_drop:.4f}")
print(f"- MSE 增加: {mse_increase:.4f}")
print("- 使用前3个特征的模型性能有所下降，但仍能捕捉主要房价影响因素。")
