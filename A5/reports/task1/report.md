# Task 1: MLP Autoencoder

## 1. 任务目标

本任务目标是为 CIFAR-10 构建一个基于 MLP 的 autoencoder，并使用一个 4 维的 embedding / latent space 对图像进行压缩与重建。

要求包括：

- 使用 MLP 架构完成编码器和解码器
- latent dimension = 4
- 训练后能够对图像进行重建
- 已生成 800 张随机样本的二维嵌入散点图，并按类别标签着色

## 2. 模型结构

模型实现位于 `src/models/task1_mlp_autoencoder.py`。

### Encoder
- 输入：`3 x 32 x 32`
- Flatten
- `Linear(3072 -> 1024)` + ReLU
- `Linear(1024 -> 256)` + ReLU
- `Linear(256 -> 64)` + ReLU
- `Linear(64 -> 4)`

### Decoder
- `Linear(4 -> 64)` + ReLU
- `Linear(64 -> 256)` + ReLU
- `Linear(256 -> 1024)` + ReLU
- `Linear(1024 -> 3072)` + Sigmoid
- Reshape back to `3 x 32 x 32`

## 3. 训练设置

训练脚本位于 `scripts/train_task1.py`。

- Dataset: CIFAR-10
- Data loader: `src/data/cifar10.py`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Batch size: `128` by default
- Latent dimension: `4`
- Device: 自动检测 `cuda` / `cpu`
- 输出目录：`reports/task1/`
  - `data/` 存放 checkpoint 和训练历史
  - `figures/` 存放重建可视化图片

## 4. 当前结果

我已经完成了 smoke test，并进一步完成了正式训练。

### Smoke test 命令

```powershell
python .\scripts\train_task1.py --epochs 1 --batch-size 64 --max-train-batches 2 --max-test-batches 1
```

### 结果

- `train_loss = 0.056055`
- `test_loss = 0.063331`

### 正式训练

正式训练在完整训练集和测试集上进行，命令如下：

```powershell
python .\scripts\train_task1.py --epochs 3 --batch-size 256 --seed 42
```

### 正式训练结果

- `epoch 1: train_loss = 0.038540, test_loss = 0.032669`
- `epoch 2: train_loss = 0.030928, test_loss = 0.030403`
- `epoch 3: train_loss = 0.030367, test_loss = 0.030224`

### 可视化结果

生成了原图与重建图预览：

- `reports/task1/figures/task1_reconstruction_preview.png`

同时生成了 loss 曲线：

- `reports/task1/figures/task1_loss_curve.png`

此外，完成了 800 张样本的二维嵌入散点图，可用于观察类别聚类效果：

- `reports/task1/figures/task1_embedding_scatter.png`

### 散点图生成步骤与原因

1. 先从已经训练好的 `Task 1` checkpoint 读取模型参数，因为这一步的目标是分析训练结果，而不是重新训练模型。
2. 再从 CIFAR-10 test set 中随机抽取 800 张样本，因为作业明确要求用随机子集来观察类别聚类效果，并且 test set 更能反映泛化后的表示空间。
3. 然后把每张图像送入 encoder 得到 4 维 latent 表示，并用 PCA 投影到二维，因为原始 embedding 是 4 维，而作业要求在二维平面中展示。
4. 接着按照类别标签着色并绘图，因为颜色编码能直接显示同类样本是否聚在一起，从而便于观察 class clustering。
5. 最后把图像和中间结果保存到 `reports/task1/figures/` 和 `reports/task1/data/`，因为这样可以保证报告中的结论可复现，也方便后续检查。

## 5. 结果文件位置

### 模型与训练历史

- `reports/task1/data/task1_mlp_autoencoder.pt`
- `reports/task1/data/task1_history.json`

### 图片

- `reports/task1/figures/task1_reconstruction_preview.png`
- `reports/task1/figures/task1_loss_curve.png`
- `reports/task1/figures/task1_embedding_scatter.png`

### 数据

- `reports/task1/data/task1_embedding_scatter.npz`
- `reports/task1/data/task1_embedding_scatter.json`

## 6. 后续可扩展内容

后续可以继续做以下扩展：

1. 使用更多 epoch 进行正式训练
2. 将 Task 1 的 MLP Autoencoder 与 Task 2 的 CNN Autoencoder 进行对比
3. 尝试 t-SNE / UMAP 等方法进一步可视化 embedding
4. 尝试不同 latent dimension 并观察 test error 的变化
5. 补充更完整的定量指标和实验分析

