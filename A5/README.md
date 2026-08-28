# CIFAR-10 Autoencoder

这是一个用于完成 CIFAR-10 Autoencoder 作业的项目。

## 默认技术栈

- Python 3.10+（推荐）
- PyTorch
- torchvision
- numpy / matplotlib / tqdm

如果课程明确要求 TensorFlow/Keras，也可以后续切换；当前先按 PyTorch 方案推进。

## 你需要先准备什么

1. 安装 Python 3.10 或更高版本。
2. 建议创建独立虚拟环境：`venv`、`conda` 都可以。
3. 安装依赖：`pip install -r requirements.txt`
4. 准备 CIFAR-10 数据集：
   - 我们后面会用 `torchvision` 自动下载；
   - 如果下载失败，也可以手动放到项目的 `data/` 目录中。
5. 确认是否有 GPU：
   - 有 CUDA 就用 GPU 训练；
   - 没有也能在 CPU 上跑通，只是更慢。

## 计划中的任务拆分

1. 环境准备与项目骨架
2. 数据加载与可视化检查
3. MLP Autoencoder（4 维嵌入）
4. CNN Autoencoder（4 维嵌入）
5. 测试集评估与对比
6. 嵌入空间距离分析
7. 不同嵌入维度的实验
8. 结果整理与报告撰写

## 运行流程（后续会补全）

- 训练脚本
- 评估脚本
- 嵌入可视化脚本
- 实验结果输出到 `outputs/`

## 第 3 步：数据加载与可视化检查

在项目根目录下运行：

```powershell
python .\scripts\check_cifar10.py
```

预期结果：

- 控制台打印训练集和测试集 batch 形状
- 打印若干标签和对应类别名
- 在 `outputs/cifar10_batch_preview.png` 生成一张 CIFAR-10 预览图

如果你只想在缺失数据时自动下载，可以加上：

```powershell
python .\scripts\check_cifar10.py --download
```


