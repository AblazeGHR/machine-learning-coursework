import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. 加载数据
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# 2. Task 1: 绘制 KDE (注意设置 bw_adjust)
sns.kdeplot(df['MedInc'], bw_adjust=0.4) # 调整带宽参数
plt.title('Median Income Distribution')
plt.savefig('pictures/Median Income Distribution', dpi=150, bbox_inches='tight')
plt.show()

# 3. Task 2 & 3: 检查双峰与偏态
# 绘制箱线图或小提琴图检查异常值
sns.boxplot(x=df['Longitude'])
plt.savefig('pictures/Longitude', dpi=150, bbox_inches='tight')

plt.show()

# 4. Task 4: 计算相关性热力图
plt.figure(figsize=(15, 10))
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.tight_layout()
plt.savefig('pictures/Median Income Distribution', dpi=150, bbox_inches='tight')
plt.show()
# 看哪个特征与 MedHouseVal 相关性最高