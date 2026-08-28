# Task 4: Latent Dimension Sweep for the CNN Autoencoder

## 1. 任务目标

本任务的目标是重复 Task 2 的 CNN autoencoder 实验，但改变 embedding / latent dimension，观察 test error 如何随着 latent dimension 的变化而变化。

本任务主要回答以下问题：

- latent dimension 增大时，重建误差是否下降？
- 当前实验设置下，哪个 latent dimension 表现最好？
- 4 维 embedding 与更大维度相比有什么差异？

## 2. 实验设置

本任务基于 Task 2 的 CNN autoencoder 架构，使用相同的数据管线与训练方式，仅改变 latent dimension。

### 模型

- 模型：`src/models/task2_cnn_autoencoder.py` 中的 `CNNAutoencoder`
- 损失函数：`MSELoss`
- 优化器：`Adam`
- 学习率：`1e-3`
- Batch size：`256`
- Epochs：`3`
- Seed：`42`

### 扫描的 latent dimension

本次实验扫描了以下维度：

- `1`
- `2`
- `4`
- `8`
- `16`

### 输出脚本

实验脚本位于 `scripts/train_task4.py`。

## 3. 实验结果

### 结果汇总表

| latent_dim | final_train_loss | final_test_loss |
|-----------:|-----------------:|----------------:|
| 1  | 0.0433448335 | 0.0430095438 |
| 2  | 0.0368676088 | 0.0368811189 |
| 4  | 0.0304337269 | 0.0302826319 |
| 8  | 0.0246011597 | 0.0244705736 |
| 16 | 0.0198395957 | 0.0183803936 |

### 结果观察

可以看到，随着 latent dimension 增大，test loss 明显下降，说明更大的 embedding space 能保留更多信息，从而提高重建质量。

其中：

- `latent_dim=1` 的 test loss 最高，重建能力最弱
- `latent_dim=16` 的 test loss 最低，为当前实验设置下的最佳结果
- `latent_dim=4` 的结果与 Task 2 / Task 3 的基线一致，处在中间水平

### 误差变化趋势

从 `1 -> 2 -> 4 -> 8 -> 16`，final test loss 近似单调下降：

- `0.0430 -> 0.0369 -> 0.0303 -> 0.0245 -> 0.0184`

这表明 latent dimension 是影响 autoencoder 重建质量的重要因素。

### 与 Task 2 / Task 3 的关系

- Task 2 固定使用 4 维 latent space
- Task 3 基于 4 维 embedding 分析类别间距离
- Task 4 进一步说明：4 维已经具有一定结构性，但若目标是更低的重建误差，更大的 latent dimension 会更有优势

## 4. 结果文件位置

### 数据文件

- `reports/task4/data/task4_sweep_config.json`
- `reports/task4/data/task4_sweep_results.csv`
- `reports/task4/data/task4_sweep_results.json`
- `reports/task4/data/task4_latent_1.pt`
- `reports/task4/data/task4_latent_2.pt`
- `reports/task4/data/task4_latent_4.pt`
- `reports/task4/data/task4_latent_8.pt`
- `reports/task4/data/task4_latent_16.pt`
- `reports/task4/data/task4_latent_1_history.json`
- `reports/task4/data/task4_latent_2_history.json`
- `reports/task4/data/task4_latent_4_history.json`
- `reports/task4/data/task4_latent_8_history.json`
- `reports/task4/data/task4_latent_16_history.json`

### 图片文件

- `reports/task4/figures/task4_test_error_vs_latent_dim.png`
- `reports/task4/figures/task4_loss_curves.png`
- `reports/task4/figures/task4_latent_1_loss_curve.png`
- `reports/task4/figures/task4_latent_2_loss_curve.png`
- `reports/task4/figures/task4_latent_4_loss_curve.png`
- `reports/task4/figures/task4_latent_8_loss_curve.png`
- `reports/task4/figures/task4_latent_16_loss_curve.png`
- `reports/task4/figures/task4_latent_4_reconstruction_preview.png`
- `reports/task4/figures/task4_latent_16_reconstruction_preview.png`

## 5. 结论

本次实验表明，CNN autoencoder 的重建误差会随着 latent dimension 的增大而持续下降。

在当前设置下：

- 最优 latent dimension 是 `16`
- 对应 final test loss 为 `0.0183803936`
- 与 `latent_dim=1` 相比，test loss 有显著下降

因此，如果优先目标是最小化重建误差，较大的 embedding 维度更有利；但如果希望获得更紧凑、更便于分析的 latent space，则 4 维依然具有研究价值。

## 6. 后续可扩展内容

后续可以继续做以下扩展：

1. 扫描更多 latent dimension，例如 `32` 或 `64`
2. 比较不同 latent dimension 下的重建图像质量
3. 结合 Task 3 的 embedding 距离分析，比较不同维度的可解释性
4. 进一步研究重建误差与类别可分性之间的关系

