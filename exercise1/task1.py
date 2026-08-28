
from commom_import import *
df,data,features=init_all()

# 2. 定义特征列表（对应图片里的8个特征）
# 3. 绘图
plt.figure(figsize=(16, 10))
for i, feat in enumerate(features):
    plt.subplot(2, 4, i + 1)
    # 加个容错：确保列存在再画
    if feat in df.columns:
        sns.kdeplot(df[feat], bw_method="scott")
        plt.title(feat)
    else:
        print(f"警告：特征 {feat} 不存在于数据集中")

plt.tight_layout()
sava_png(__file__,'KDE_Distribution')
plt.show()