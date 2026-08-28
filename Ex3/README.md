# Ex3: Titanic 集成学习实验

本项目完成以下任务：
- 使用 Titanic 数据集训练 **随机森林** 模型预测生还情况。
- 使用 **逻辑回归** 作为基线模型。
- 采用训练/测试划分 + 5 折交叉验证调参。
- 在测试集上对比结果，并给出“随机森林相对逻辑回归提升了多少”的直观结论。

## 1. 项目结构

- `src/main.py`：主入口，执行训练、调参与评估
- `src/data_utils.py`：数据加载与字段统一
- `src/modeling.py`：预处理和模型管线定义
- `src/reporting.py`：保存指标、图表、Markdown 报告
- `requirements.txt`：依赖
- `outputs/`：运行后自动生成（指标、参数、图）
- `reports/`：运行后自动生成（中文实验报告）

## 2. 数据说明

优先读取本地文件：
- `data/raw/train.csv`（Kaggle Titanic 训练集）

如果本地没有该文件，程序会尝试从 OpenML 下载 Titanic 数据并缓存到：
- `data/raw/titanic_openml.csv`

## 3. 快速运行（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.main
```

## 4. 输出结果

运行后可查看：
- `outputs/metrics.csv`：两个模型的测试集指标
- `outputs/best_params.json`：CV 最优参数
- `outputs/predictions_test.csv`：逐样本预测结果
- `outputs/figures/`：混淆矩阵与 ROC 曲线图
- `reports/实验报告.md`：中文结论报告

## 5. 任务对应说明

`reports/实验报告.md` 中会给出类似下面的结论：
- 随机森林相对逻辑回归在 Accuracy 上提升/下降了多少百分点。
- 随机森林相对逻辑回归在 F1 上提升/下降了多少百分点。

这正对应作业要求中的：
- “Train a random forest...”
- “How much better is your prediction compared to a logistic regression?”

