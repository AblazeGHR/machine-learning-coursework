# CIFAR-10 Autoencoder Final Report

## 1. 项目总览

本项目围绕 CIFAR-10 数据集完成了四个阶段的 autoencoder 实验：

- Task 1：MLP Autoencoder，latent dimension = 4
- Task 2：CNN Autoencoder，latent dimension = 4
- Task 3：分析 CNN autoencoder 的 embedding space 中不同类别之间的距离
- Task 4：扫描不同 latent dimension，分析 test error 随维度变化的趋势

### 总体结论

- Task 1 和 Task 2 都完成了完整的重建训练流程，并能稳定输出重建图与 loss 曲线
- Task 1 额外完成了 800 张样本的二维嵌入散点图，用于观察类别聚类效果
- Task 3 说明 4 维 embedding 已经具有一定语义结构，可用于类别距离分析
- Task 4 表明 latent dimension 越大，重建误差越低；当前 sweep 下 `latent_dim=16` 表现最好

## 2. 实验设置

### 共同设置

- Dataset: CIFAR-10
- Loss: MSELoss
- Optimizer: Adam
- Learning rate: `1e-3`
- Random seed: `42`
- 训练与测试均基于项目中的 `src/data/cifar10.py`

### 任务区分

| Task | Model | Latent dim | 主要目的 |
|---|---|---:|---|
| Task 1 | MLP Autoencoder | 4 | 验证 MLP 重建能力 |
| Task 2 | CNN Autoencoder | 4 | 与 Task 1 对比重建性能 |
| Task 3 | CNN Embedding Analysis | 4 | 分析类别间距离与可解释性 |
| Task 4 | CNN Sweep | 1 / 2 / 4 / 8 / 16 | 分析 test error 随维度变化 |

## 3. Task 1: MLP Autoencoder

### 结果概述

Task 1 采用 MLP 编码器/解码器，latent dimension 为 4。

正式训练结果：

- Epoch 1: train loss = `0.038540`, test loss = `0.032669`
- Epoch 2: train loss = `0.030928`, test loss = `0.030403`
- Epoch 3: train loss = `0.030367`, test loss = `0.030224`

### 散点图生成步骤与原因

1. 先加载已经训练好的 Task 1 checkpoint，因为这里的目标是分析训练结果，而不是重新训练模型。
2. 再从 CIFAR-10 test set 中随机抽取 800 张图片，因为作业要求使用随机子集，并且 test set 更适合观察模型学到的表示是否具有泛化性。
3. 然后把图片送入 encoder 得到 4 维 latent 表示，并用 PCA 投影到二维，因为作业要求把 embedding 映射到二维平面来观察聚类效果。
4. 接着按照类别标签着色并绘图，因为颜色编码可以直接显示同类样本是否聚集在一起。
5. 最后保存图像和中间数据到 `reports/task1/`，因为这样可以让结果可复现，也方便在报告中引用。

### 图片留白

- [图片留白：Task 1 原图 vs 重建图]
- [图片留白：Task 1 loss curve]
- [图片留白：Task 1 800 张样本的二维嵌入散点图]

### 结果文件

- `reports/task1/data/task1_mlp_autoencoder.pt`
- `reports/task1/data/task1_history.json`
- `reports/task1/data/task1_embedding_scatter.npz`
- `reports/task1/data/task1_embedding_scatter.json`
- `reports/task1/figures/task1_reconstruction_preview.png`
- `reports/task1/figures/task1_loss_curve.png`
- `reports/task1/figures/task1_embedding_scatter.png`

## 4. Task 2: CNN Autoencoder

### 结果概述

Task 2 采用 CNN 编码器/解码器，latent dimension 为 4。

正式训练结果：

- Epoch 1: train loss = `0.039766`, test loss = `0.030985`
- Epoch 2: train loss = `0.030711`, test loss = `0.030491`
- Epoch 3: train loss = `0.030431`, test loss = `0.030284`

### 与 Task 1 的对比

- Task 1 最终 test loss：`0.030224`
- Task 2 最终 test loss：`0.030284`

两者非常接近，说明在当前设置下，MLP Autoencoder 和 CNN Autoencoder 的重建误差处于同一量级。

### 图片留白

- [图片留白：Task 2 原图 vs 重建图]
- [图片留白：Task 2 loss curve]

### 结果文件

- `reports/task2/data/task2_cnn_autoencoder.pt`
- `reports/task2/data/task2_history.json`
- `reports/task2/figures/task2_reconstruction_preview.png`
- `reports/task2/figures/task2_loss_curve.png`

## 5. Task 3: Embedding Distance Analysis

### 结果概述

Task 3 使用 Task 2 的 4 维 embedding，对 CIFAR-10 test set 的类别中心进行距离分析。

关键结果：

- 全局最近类别对：`deer` 和 `frog`，距离 `0.4133`
- 全局最远类别对：`airplane` 和 `frog`，距离 `5.4487`
- embedding 维度：`4`
- 分析样本数：`10000`

### 结论

4 维 embedding 已经能表现出一定的类别结构：语义相近的类别更接近，语义差异大的类别距离更远。

### 图片留白

- [图片留白：Task 3 距离热力图]
- [图片留白：Task 3 类别中心距离柱状图]

### 结果文件

- `reports/task3/data/task3_embeddings.npz`
- `reports/task3/data/task3_class_centers.json`
- `reports/task3/data/task3_distance_matrix.csv`
- `reports/task3/data/task3_distance_summary.json`
- `reports/task3/data/task3_config.json`
- `reports/task3/figures/task3_distance_heatmap.png`
- `reports/task3/figures/task3_center_distance_bar.png`

## 6. Task 4: Latent Dimension Sweep

### 结果概述

Task 4 扫描 latent dimension = `1, 2, 4, 8, 16`，并比较最终 test loss。

| latent_dim | final_test_loss |
|---:|---:|
| 1  | 0.0430095438 |
| 2  | 0.0368811189 |
| 4  | 0.0302826319 |
| 8  | 0.0244705736 |
| 16 | 0.0183803936 |

### 结论

- latent dimension 越大，test error 越低
- 当前 sweep 下最优维度是 `16`
- `latent_dim=4` 仍然是一个兼顾紧凑性与可解释性的中间点

### 图片留白

- [图片留白：Task 4 test error vs latent dimension]
- [图片留白：Task 4 不同维度 loss curves]
- [图片留白：Task 4 latent_dim=4 重建图]
- [图片留白：Task 4 latent_dim=16 重建图]

### 结果文件

- `reports/task4/data/task4_sweep_config.json`
- `reports/task4/data/task4_sweep_results.csv`
- `reports/task4/data/task4_sweep_results.json`
- `reports/task4/data/task4_latent_1.pt`
- `reports/task4/data/task4_latent_2.pt`
- `reports/task4/data/task4_latent_4.pt`
- `reports/task4/data/task4_latent_8.pt`
- `reports/task4/data/task4_latent_16.pt`
- `reports/task4/figures/task4_test_error_vs_latent_dim.png`
- `reports/task4/figures/task4_loss_curves.png`

## 7. 综合比较

| Task | 核心问题 | 关键发现 |
|---|---|---|
| Task 1 | MLP 是否能重建 CIFAR-10？ | 可以，且 loss 收敛稳定 |
| Task 2 | CNN 是否优于 MLP？ | 两者误差接近，CNN 并未明显胜出 |
| Task 3 | 4 维 embedding 是否有结构？ | 有，类别之间存在可解释距离关系 |
| Task 4 | latent dim 如何影响误差？ | 越大越好，16 维最佳 |

## 8. 总结

本项目完成了从重建、对比、表示分析到维度扫描的完整 autoencoder 实验链条。

最终可以得到以下结论：

1. autoencoder 可以有效压缩并重建 CIFAR-10 图像
2. CNN 与 MLP 在当前设置下的重建性能相近
3. 4 维 latent space 已经具备一定可解释性
4. 若目标是最小化重建误差，更大的 latent dimension 更合适

## 9. 附录：主要文件索引

- `reports/task1/report.md`
- `reports/task2/report.md`
- `reports/task3/report.md`
- `reports/task4/report.md`

