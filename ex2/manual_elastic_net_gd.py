import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error

# 复用 Task 1 的最优参数
alpha = 0.01
l1_ratio = 0.1

# 1. 加载 California Housing 数据集
data = fetch_california_housing()
X = data.data
y = data.target

# 进行 80% 训练集 / 20% 测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 对特征做标准化预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置列到 X (b 作为 w 的最后一个元素)
X_train_scaled = np.hstack([X_train_scaled, np.ones((X_train_scaled.shape[0], 1))])
X_test_scaled = np.hstack([X_test_scaled, np.ones((X_test_scaled.shape[0], 1))])

# 3. 手动实现 Elastic-Net 梯度下降
def elastic_net_loss(X, y, w, alpha, l1_ratio):
    n = len(y)
    y_pred = X @ w
    mse_loss = (1 / (2 * n)) * np.sum((y - y_pred) ** 2)
    l1_penalty = alpha * l1_ratio * np.sum(np.abs(w[:-1]))  # L1 只对权重，不对偏置
    l2_penalty = 0.5 * alpha * (1 - l1_ratio) * np.sum(w[:-1] ** 2)
    return mse_loss + l1_penalty + l2_penalty

def elastic_net_grad(X, y, w, alpha, l1_ratio):
    n = len(y)
    y_pred = X @ w
    error = y - y_pred
    grad_mse = - (1 / n) * X.T @ error
    grad_l1 = np.zeros_like(w)
    grad_l1[:-1] = alpha * l1_ratio * np.sign(w[:-1])  # L1 梯度
    grad_l2 = np.zeros_like(w)
    grad_l2[:-1] = alpha * (1 - l1_ratio) * w[:-1]  # L2 梯度
    return grad_mse + grad_l1 + grad_l2

# 梯度下降参数
# (A) 使用批量梯度下降（Batch GD），理由：数据集小（~16000 训练样本），可以一次性计算梯度，收敛稳定。
# 超参数：学习率 lr=0.01，最大迭代 max_iter=10000，收敛阈值 tol=1e-6。
lr = 0.01
max_iter = 10000
tol = 1e-6

# 初始化 w (包括偏置)
w = np.zeros(X_train_scaled.shape[1])

# 记录损失
losses = []

for iter in range(max_iter):
    loss = elastic_net_loss(X_train_scaled, y_train, w, alpha, l1_ratio)
    losses.append(loss)
    
    grad = elastic_net_grad(X_train_scaled, y_train, w, alpha, l1_ratio)
    w -= lr * grad
    
    # 检查收敛
    if iter > 0 and abs(losses[-1] - losses[-2]) < tol:
        break

# (B) 收敛步数
converged_iter = len(losses)
print(f"梯度下降收敛步数: {converged_iter}")

# 4. 与 sklearn ElasticNet 对比
# 训练 sklearn 模型
model_sklearn = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
model_sklearn.fit(X_train_scaled[:, :-1], y_train)  # 不包括偏置列

# sklearn 的 coef_ 和 intercept_
w_sklearn = np.append(model_sklearn.coef_, model_sklearn.intercept_)

# 手动 GD 的 w
w_manual = w

# (C) 参数差异
param_diff = w_manual - w_sklearn
param_mse = np.mean(param_diff ** 2)
print(f"参数差异 MSE: {param_mse:.6f}")
print("手动 GD w:", w_manual)
print("sklearn w:", w_sklearn)

# 评估手动 GD 模型
y_pred_manual = X_test_scaled @ w_manual
r2_manual = r2_score(y_test, y_pred_manual)
mse_manual = mean_squared_error(y_test, y_pred_manual)
print(f"手动 GD R²: {r2_manual:.4f}")
print(f"手动 GD MSE: {mse_manual:.4f}")

# 评估 sklearn 模型
y_pred_sklearn = model_sklearn.predict(X_test_scaled[:, :-1])
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(f"sklearn R²: {r2_sklearn:.4f}")
print(f"sklearn MSE: {mse_sklearn:.4f}")

# 5. 输出损失变化曲线
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Elastic-Net Loss during Gradient Descent')
plt.savefig('elastic_net_loss_curve.png')
plt.show()

print("手动 Elastic-Net 梯度下降完成。")
