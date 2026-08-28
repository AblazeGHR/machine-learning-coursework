import numpy as np
from scipy import stats


def calculate_scott_bandwidth_scipy(data):
    x = np.asarray(data)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("样本数必须大于等于2")

    # 1. 计算总体标准差 (ddof=0)
    std = x.std(ddof=0)
    if std == 0:
        return 0.0

    # 2. 标准化数据后计算IQR (scipy 内部逻辑)
    x_norm = x / std
    q75, q25 = np.percentile(x_norm, [75, 25])
    iqr_norm = q75 - q25

    # 3. 计算鲁棒标准差
    sigma = min(1.0, iqr_norm / 1.349) * std

    # 4. Scott 公式
    scott_bw = 1.059 * sigma * (n ** (-1 / 5))
    return scott_bw


def calculate_scott_bandwidth_simple(data):
    """
    简化版 Scott 带宽计算（无鲁棒性修正，仅核心公式）
    """
    x = np.asarray(data)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 2:
        raise ValueError("数据点数量必须大于等于2")

    # 仅用总体标准差（无IQR鲁棒修正）
    std = x.std(ddof=0)  # 总体标准差，和scipy一致
    if std == 0:
        return 0.0

    # 核心公式（无鲁棒调整）
    scott_bw_simple = 1.059 * std * (n ** (-1 / 5))
    return scott_bw_simple
# 测试验证
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

my_bw = calculate_scott_bandwidth_scipy(data)
kde = stats.gaussian_kde(data, bw_method='scott')
sci_bw = kde.factor
simple_bw = calculate_scott_bandwidth_simple(data)

print(f"修正后自己计算的Scott带宽: {my_bw:.6f}")
print(f"scipy的Scott带宽: {sci_bw:.6f}")
print(f"简化版本：{simple_bw:.6f}")
print(f"结果是否一致: {np.isclose(my_bw, sci_bw)}")