from commom_import import *
df,data,_=init_all()

# ===================== 1. 缺失值 + 异常值概览 =====================
# 创建1行2列的子图，设置画布大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 图1：缺失值热力图（红色=缺失，白色=无缺失）
sns.heatmap(df.isnull(), cmap='Reds', cbar=False, ax=ax1)
ax1.set_title('Missing Values Distribution', fontsize=14)  # 英文标题
ax1.set_xlabel('Features', fontsize=12)
ax1.set_ylabel('Samples', fontsize=12)

# 图2：数值特征箱线图（识别异常值）
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.boxplot(data=df[num_cols], ax=ax2)
ax2.set_title('Boxplot of Numeric Features (Outlier Detection)', fontsize=14)
ax2.set_xlabel('Numeric Features', fontsize=12)
ax2.tick_params(axis='x', rotation=45)  # X轴标签旋转，避免重叠

plt.tight_layout()  # 自动调整布局
sava_png(__file__,"missing_outlier_analysis")
plt.show()

# ===================== 2. 数值特征分布 =====================
# 筛选数值列，自动计算子图行数（每列2个特征）
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
n_rows = (len(num_cols) + 1) // 2  # 向上取整，避免多余空白

# 创建子图
fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
axes = axes.flatten()  # 展平轴，方便循环

# 逐个绘制直方图+核密度图
for i, col in enumerate(num_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Count', fontsize=10)

# 隐藏多余的子图（如果特征数为奇数）
if len(num_cols) < len(axes):
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
sava_png(__file__,'numeric_distribution')
plt.show()

# ===================== 3. 特征相关性热力图 =====================
# 计算数值特征的相关系数矩阵
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,  # 显示相关系数数值
    annot_kws={'size': 9},  # 数值字体大小
    cmap='RdBu_r',  # 正红负蓝配色
    vmin=-1, vmax=1,  # 固定相关性范围（-1到1）
    square=True,  # 正方形格子
    linewidths=0.5  # 格子间黑线，增强区分度
)
plt.title('Correlation Matrix of Features', fontsize=16)
plt.xticks(rotation=45)  # X轴标签旋转
plt.tight_layout()
sava_png(__file__,'correlation_matrix')
plt.show()

# ===================== 4. 类别特征分析（如有） =====================
# 筛选类别列（object类型）
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图1：类别计数
    sns.countplot(x=cat_cols[0], data=df, ax=ax1)
    ax1.set_title(f'Category Count of {cat_cols[0]}', fontsize=14)
    ax1.set_xlabel(cat_cols[0], fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # 图2：类别对目标变量的影响（以房价为例）
    target_col = 'MedHouseVal'  # 目标列名（可替换）
    sns.boxplot(x=cat_cols[0], y=target_col, data=df, ax=ax2)
    ax2.set_title(f'Impact of {cat_cols[0]} on {target_col}', fontsize=14)
    ax2.set_xlabel(cat_cols[0], fontsize=12)
    ax2.set_ylabel(target_col, fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    sava_png(__file__,'categorical_analysis')
    plt.show()

# ===================== 5. 特征与目标变量关系 =====================
# 定义目标列（可根据你的需求替换）
target_col = MedHouseVal
# 筛选数值特征（排除目标列）
num_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != target_col]

# 计算子图行数
n_rows = (len(num_cols) + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
axes = axes.flatten()

# 逐个绘制散点图+回归线（看线性关系）
for i, col in enumerate(num_cols):
    sns.regplot(x=col, y=target_col, data=df, ax=axes[i], scatter_kws={'alpha': 0.3})
    axes[i].set_title(f'{col} vs {target_col}', fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel(target_col, fontsize=10)

# 隐藏多余子图
if len(num_cols) < len(axes):
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
sava_png(__file__,'feature_target_relationship')
plt.show()