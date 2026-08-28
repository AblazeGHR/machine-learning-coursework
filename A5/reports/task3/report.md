# Task 3: Embedding Distance Analysis of the CNN Autoencoder

## 1. 任务目标

本任务的目标是分析 Task 2 中训练好的 CNN autoencoder 在 embedding space 里的类别分布情况，测量 CIFAR-10 不同类别之间的距离，并回答以下问题：

- 哪些类别在 embedding space 中最接近？
- 哪些类别在 embedding space 中最远？
- 不同类别之间的距离矩阵长什么样？

本任务不重新训练模型，而是直接使用 Task 2 的 checkpoint 进行分析。

## 2. 方法说明

本任务使用 `reports/task2/data/task2_cnn_autoencoder.pt` 作为已训练好的模型参数，加载 `src/models/task2_cnn_autoencoder.py` 中的 `CNNAutoencoder`，并调用其 `encode()` 方法提取 CIFAR-10 test set 上每张图片的 4 维 latent 表示。

分析步骤如下：

1. 使用 CIFAR-10 test set 作为分析对象
2. 提取每个样本的 4 维 embedding
3. 按类别计算 embedding 均值，得到每个类别的中心点
4. 计算类别中心之间的欧氏距离，形成 10 × 10 距离矩阵
5. 找出全局最近类别对和最远类别对
6. 统计每个类别最近和最远的其他类别

分析脚本位于 `scripts/analyze_task3.py`。

## 3. 结果概述

本次分析一共处理了 `10000` 个测试样本，embedding 维度为 `4`，一共包含 `10` 个类别。

### 全局最近类别对

- `deer` 和 `frog`
- 距离：`0.4133`

### 全局最远类别对

- `airplane` 和 `frog`
- 距离：`5.4487`

### 类别距离矩阵观察

从距离矩阵和热力图可以看出：

- `deer`、`frog`、`cat`、`dog` 之间的距离普遍较小，说明这些类别在 embedding space 里更接近。
- `airplane` 与大多数类别距离都较大，尤其与 `frog` 最远。
- `ship` 和 `truck` 的距离也较小，说明这两个类别在 latent space 中比较接近。

## 4. 结果文件位置

### 数据文件

- `reports/task3/data/task3_embeddings.npz`
- `reports/task3/data/task3_class_centers.json`
- `reports/task3/data/task3_distance_matrix.csv`
- `reports/task3/data/task3_distance_summary.json`
- `reports/task3/data/task3_config.json`

### 图片文件

- `reports/task3/figures/task3_distance_heatmap.png`
- `reports/task3/figures/task3_center_distance_bar.png`

## 5. 结论

Task 3 说明：CNN autoencoder 学到的 4 维 embedding 并不是完全随机的，而是已经在一定程度上把语义相近的类别映射到了更接近的位置。

尤其是：

- `deer` 和 `frog` 最接近
- `airplane` 和 `frog` 最远

这说明 latent space 中的类别结构具有一定可解释性，可以作为后续更深入分析的基础。

## 6. 后续可扩展内容

后续可以继续做以下扩展：

1. 用 PCA 或 t-SNE 可视化 embedding
2. 比较 train set 和 test set 上的类别中心差异
3. 将 Task 3 的距离分析与 Task 1 的 embedding 结果对比
4. 在更高维或不同维度的 latent space 上重复分析

