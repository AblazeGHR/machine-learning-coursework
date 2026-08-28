import warnings
# 导入所需库
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from scipy.stats import gaussian_kde

import os
MedHouseVal = 'MedHouseVal'
warnings.filterwarnings('ignore')

def base_init():
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style('whitegrid')  # 设置图表样式

def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data

def init_all():
    base_init()
    df, data = load_data()
    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    return df, data,features
def sava_png(filename, feat_name):
    file_name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(f'pictures/{file_name_no_ext}', exist_ok=True)
    plt.savefig(f"pictures/{file_name_no_ext}/{feat_name}.png", dpi=150, bbox_inches='tight')
if __name__ == "__main__":
    import sklearn.datasets as datasets
    # 查看数据集根目录（核心代码）
    data_path = datasets.get_data_home()
    print(f"sklearn数据集默认存储路径：\n{data_path}")
    data = fetch_california_housing()
    df = pd.DataFrame(
        data=data.data,  # 特征列
        columns=data.feature_names  # 特征名
    )
    df[data.target_names[0]] = data.target  # 把房价作为新列加入
    df.to_csv('data.csv', index=False, encoding='utf-8-sig')
    df['MedHouseVal'] = data.target





