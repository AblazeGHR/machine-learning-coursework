import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

from commom_import import sava_png, init_all

df,data,features=init_all()

# Load California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Define heavy-tailed features to transform
tail_features = ["AveRooms", "AveBedrms", "Population", "AveOccup"]


# Define a function to compare original and transformed distributions
def plot_compare(feat, original_data, transformed_data, transform_type):
    plt.figure(figsize=(12, 4))

    # Plot original distribution
    plt.subplot(1, 2, 1)
    sns.kdeplot(original_data, bw_method="scott")
    plt.title(f"{feat} - Original (Heavy Tail)")
    plt.xlabel(feat)
    plt.ylabel("Probability Density")

    # Plot transformed distribution
    plt.subplot(1, 2, 2)
    sns.kdeplot(transformed_data, bw_method="scott")
    plt.title(f"{feat} - After {transform_type} Transformation")
    plt.xlabel(f"{feat} ({transform_type})")
    plt.ylabel("Probability Density")

    plt.tight_layout()
    sava_png(__file__,f"{feat} - After {transform_type} Transformation")
    plt.show()


# Apply Log1p transformation (log(x+1) to avoid log(0))
for feat in tail_features:
    if feat == "AveOccup":
        # Truncate extreme values (99th quantile) before log transformation for AveOccup
        threshold = df[feat].quantile(0.99)
        truncated_data = df[feat].clip(upper=threshold)
        df[f"{feat}_log1p"] = np.log1p(truncated_data)
        plot_compare(feat, df[feat], df[f"{feat}_log1p"], "Log1p (Truncated)")
    else:
        # Direct Log1p transformation for other features
        df[f"{feat}_log1p"] = np.log1p(df[feat])
        plot_compare(feat, df[feat], df[f"{feat}_log1p"], "Log1p")

# Optional: Save transformed data to CSV (if needed)
# df.to_csv("california_housing_log_transformed.csv", index=False)