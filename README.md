# 机器学习课程作业存档（MIS110）

个人机器学习课程（MIS110）作业与练习的存档仓库。数据集不入库，下载方式见下。

## 目录结构

| 目录 | 内容 |
|------|------|
| `exercise1/` | 练习 1：加州房价数据探索与 KDE 分析（含题目 `1.pdf`、LaTeX 源码 `.tmu`、提交报告 PDF） |
| `ex2/` | 练习 2：房价回归（线性回归基线、Elastic Net 手写梯度下降、特征重要性分析） |
| `Ex3/` | 练习 3：Titanic 生存预测（逻辑回归 vs 随机森林，网格搜索，含 `requirements.txt`） |
| `A4/` | 作业 4：UCI HAR 人体活动识别 |
| `A5/` | 作业 5：CIFAR-10（MLP 自编码器等任务），详见 `A5/README.md` |
| `A6/` | 作业 6：脑卒中 TCAV 概念激活向量可解释性分析（PyTorch），详见 `A6/README.md` |

## 数据集下载

- **A4（UCI HAR Dataset）**：从 [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) 下载 zip，解压到 `A4/data/`。
- **A5（CIFAR-10）**：`A5` 脚本会自动通过 torchvision 下载到 `A5/data/`，或手动下载 [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 放入 `A5/data/`。
- **Ex3（Titanic）**：数据已随仓库提供（`Ex3/data/raw/titanic_openml.csv`）。

## 环境

各作业独立运行，Python ≥ 3.9；`Ex3` 与 `A6` 提供 `requirements.txt`，其余作业依赖 numpy / pandas / scikit-learn / matplotlib / torch，按需安装。
