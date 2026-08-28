# Task 2: CNN Autoencoder

## 1. 任务目标

本任务目标是在 CIFAR-10 数据集上构建一个基于 CNN 的 autoencoder，并使用 4 维 latent space 对图像进行压缩与重建。

要求包括：

- 使用 CNN 架构完成编码器和解码器
- latent dimension = 4
- 训练后能够重建图像
- 与 Task 1 的 MLP Autoencoder 进行对比

## 2. 模型结构

模型实现位于 `src/models/task2_cnn_autoencoder.py`。

### Encoder
- 输入：`3 x 32 x 32`
- `Conv2d(3 -> 32, kernel=3, stride=2, padding=1)` + ReLU
- `Conv2d(32 -> 64, kernel=3, stride=2, padding=1)` + ReLU
- `Conv2d(64 -> 128, kernel=3, stride=2, padding=1)` + ReLU
- Flatten
- `Linear(128 * 4 * 4 -> 4)`

### Decoder
- `Linear(4 -> 128 * 4 * 4)` + ReLU
- Reshape to `128 x 4 x 4`
- `ConvTranspose2d(128 -> 64, kernel=4, stride=2, padding=1)` + ReLU
- `ConvTranspose2d(64 -> 32, kernel=4, stride=2, padding=1)` + ReLU
- `ConvTranspose2d(32 -> 16, kernel=4, stride=2, padding=1)` + ReLU
- `Conv2d(16 -> 3, kernel=3, stride=1, padding=1)` + Sigmoid

## 3. 训练设置

训练脚本位于 `scripts/train_task2.py`。

- Dataset: CIFAR-10
- Data loader: `src/data/cifar10.py`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Batch size: `128` by default
- Latent dimension: `4`
- Device: automatically detects `cuda` / `cpu`
- Output directory: `reports/task2/`
  - `data/` stores checkpoints and training history
  - `figures/` stores reconstruction visualizations

## 4. 当前结果

我已经完成了 smoke test，并进一步完成了正式训练。

### Smoke Test Command

```powershell
python .\scripts\train_task2.py --epochs 1 --batch-size 64 --max-train-batches 2 --max-test-batches 1
```

### 结果

- `train_loss = 0.060081`
- `test_loss = 0.064947`

### 正式训练

正式训练在完整训练集和测试集上进行，命令如下：

```powershell
python .\scripts\train_task2.py --epochs 3 --batch-size 256 --seed 42
```

### 正式训练结果

- `epoch 1: train_loss = 0.039766, test_loss = 0.030985`
- `epoch 2: train_loss = 0.030711, test_loss = 0.030491`
- `epoch 3: train_loss = 0.030431, test_loss = 0.030284`

### 可视化结果

生成了原图与重建图预览：

- `reports/task2/figures/task2_reconstruction_preview.png`

同时生成了 loss 曲线：

- `reports/task2/figures/task2_loss_curve.png`

### 与 Task 1 的对比

在相同训练轮数、批大小和随机种子设置下，Task 1 的最终测试损失为 `0.030224`，Task 2 的最终测试损失为 `0.030284`。

两者非常接近，其中 Task 1 在这次正式训练中略优于 Task 2，但差距很小，说明 CNN Autoencoder 与 MLP Autoencoder 在当前设置下的重建误差处于同一量级。

### MSE 对比（简要）

- 我对两个模型在整个 CIFAR-10 test 集上逐张计算了重建 MSE，并把对比结果保存到了 `reports/compare/`：
  - 汇总：`reports/compare/task1_task2_mse_summary.json`
  - 每图 MSE 数据：`reports/compare/task1_task2_per_image_mse.npz`
  - 对比图：`reports/compare/figures/per_image_mse_histogram.png`、`reports/compare/figures/per_image_mse_diff_histogram.png`
- 关键数值（n=10000）：
  - MLP mean MSE = 0.03022369
  - CNN mean MSE = 0.03028437
  - 平均差（CNN - MLP）≈ 0.0000607（约 0.2% 的相对差异）
  - 逐图比较：MLP 更优 5068 张，CNN 更优 4932 张

结论：两个模型在 test 集上的 reconstruction MSE 基本一致，MLP 略优但差异微小，可能由随机性、训练轮次不足、模型结构差异或 MSE 指标对感知质量不敏感等因素造成。如需更确定的结论，建议多次 seed 重复、延长训练或使用感知度量（PSNR/SSIM/LPIPS）并可视化最差的重建样本进行人工检查。

## 5. 结果文件位置

### 模型与训练历史

- `reports/task2/data/task2_cnn_autoencoder.pt`
- `reports/task2/data/task2_history.json`

### 图片

- `reports/task2/figures/task2_reconstruction_preview.png`
- `reports/task2/figures/task2_loss_curve.png`

## 6. 后续可扩展内容

后续可以继续做以下扩展：

1. 使用更多 epoch 进行正式训练
2. 将 Task 2 与 Task 1 的重建结果进行对比
3. 在 embedding space 中分析不同类别之间的距离
4. 尝试不同 latent dimension 并观察 test error 的变化
5. 补充更完整的定量指标和更详细的实验分析

