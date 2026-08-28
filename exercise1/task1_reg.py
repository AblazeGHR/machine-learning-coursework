from scipy.signal.windows import gaussian
import scipy
from commom_import import *
df,data,features=init_all()



# Define a function to compare original and transformed distributions
def draw_and_save_plot_(feat, transformed_data, transform_type):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(transformed_data, bw_method="scott")
    kde = gaussian_kde(transformed_data, bw_method="scott")
    bandwidth = kde.factor
    plt.title(f"{feat} - After {transform_type} Transformation - bandwidth: {bandwidth:.4f}")
    plt.xlabel(f"{feat} ({transform_type})")
    plt.ylabel("Probability Density")
    plt.tight_layout()
    sava_png(__file__,feat)
    plt.show()
def clip(feat,df):
    threshold = df[feat].quantile(0.99)
    truncated_data = df[feat].clip(upper=threshold)
    transformed_feat_name= f"{feat}_log1p"
    df[f"{feat}_log1p"] = np.log1p(truncated_data)
    return transformed_feat_name
def draw_all_in_one(tail_features,gaussian_features,df):
    n_features = len(tail_features)+len(gaussian_features)
    n_rows = (n_features + 1) // 2  # 向上取整
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    # 展平axes（方便循环，避免行列索引麻烦）
    axes = axes.flatten()


    for idx, feat in enumerate(tail_features):

        transformed_feat_name=clip(feat,df)
        # draw_and_save_plot_(feat, df[f"{feat}_log1p"], "Log1p (Truncated)")
        data = df[transformed_feat_name].dropna().values.astype(np.float64)
        kde = gaussian_kde(data, bw_method="scott")
        bandwidth = kde.factor * np.std(data, ddof=1)
        ax = axes[idx]
        # 在指定子图上绘制KDE
        sns.kdeplot(data=df, x=transformed_feat_name, bw_method='scott', fill=True, color='steelblue', ax=ax)

        ax.set_title(f"{transformed_feat_name} - bandwidth: {bandwidth:.4f}", fontsize=10)
        ax.set_xlabel(f"{transformed_feat_name}")
        ax.set_ylabel('Density', fontsize=8)
        # 调整刻度字体大小
        ax.tick_params(axis='both', labelsize=8)



    for idx, feat in enumerate(gaussian_features):
        new_idx=idx+len(tail_features)
        # draw_and_save_plot_(feat, df[f"{feat}_log1p"], "Log1p (Truncated)")
        ax = axes[new_idx]

        # 在指定子图上绘制KDE
        data = df[feat].dropna().values.astype(np.float64)
        kde = gaussian_kde(data, bw_method="scott")
        bandwidth =  kde.factor * np.std(data, ddof=1)
        print(f'{bandwidth:.4f}')
        data = df[feat].dropna().values.astype(np.float64)
        sigma = np.std(data, ddof=1)  # 样本标准差
        n = len(data)  # 样本量
        true_scott_bw = 1.059 * sigma * (n ** (-1 / 5))  # Scott核心公式
        print(f'{true_scott_bw:.4f}')

        sns.kdeplot(data=df, x=feat, bw_method='scott', fill=True, color='steelblue', ax=ax)
        ax.set_title(f"{feat} - bandwidth: {bandwidth:.4f}", fontsize=10)
        ax.set_xlabel(f"{feat}")
        ax.set_ylabel('Density', fontsize=8)
        # 调整刻度字体大小
        ax.tick_params(axis='both', labelsize=8)




    # 隐藏多余的子图（如果特征数不是偶数）
    if n_features < len(axes):
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])

    # 整体标题和布局调整
    fig.suptitle('KDE Distributions of All Continuous Features (California Housing)', fontsize=14, y=1.02)
    plt.tight_layout()
    # 保存整合图
    sava_png(__file__,"All Continuous Features")
    plt.show()

tail_features = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
gaussian_features=["HouseAge","Latitude",'Longitude','MedInc']
for feat in tail_features:
    transformed_feat_name = clip(feat,df)
    draw_and_save_plot_(feat, df[f'{transformed_feat_name}'], "Log1p (Truncated)")
draw_all_in_one(tail_features,gaussian_features,df)

