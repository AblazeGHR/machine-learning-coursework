# MNIST TCAV Experiment

这个项目先训练一个 MNIST CNN，再用 TCAV 分析笔画概念对数字识别的影响。当前仓库里有三条主流程：单概念 X-shape 基线、多个笔画概念对比，以及用 PyTorch 训练 CAV 的加速版本。

## 项目内容

- `run_tcav.py`：训练 MNIST CNN，并对 X-shape 概念做 TCAV 分析，输出模型、图表和汇总。
- `run_stroke_concepts_tcav.py`：加载训练好的模型，对 6 个笔画概念做对比分析，使用传统的线性 CAV 实现。
- `run_stroke_concepts_tcav_torch_cav.py`：和上一个脚本相同的概念对比流程，但 CAV 训练改为 PyTorch 版本，并在循环里加入进度条。

## 1) 创建并激活虚拟环境（PowerShell）

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本执行，运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
\.venv\Scripts\Activate.ps1
```

## 2) 安装依赖

```powershell
pip install -r requirements.txt
```

## 3) 运行脚本

先训练并生成模型，再运行多概念分析脚本。

- 训练 MNIST CNN 并做 X-shape TCAV：

```powershell
\.venv\Scripts\python.exe -u run_tcav.py --epochs 3 --concept-size 300 --eval-size 256 --runs 10
```

- 快速 smoke 测试，适合检查环境是否可用：

```powershell
\.venv\Scripts\python.exe -u run_tcav.py --quick
```

- 运行原始多概念脚本，默认读取 `outputs/run_tcav/models/mnist_cnn.pt`：

```powershell
\.venv\Scripts\python.exe -u run_stroke_concepts_tcav.py --runs 12
```

- 运行 PyTorch CAV 版本，适合更快的 CAV 训练或 GPU 环境：

```powershell
\.venv\Scripts\python.exe -u run_stroke_concepts_tcav_torch_cav.py --runs 20 --concept-size 400 --eval-size 512 --cav-epochs 40
```

## 4) 输出结构

- `outputs/run_tcav/`
	- `models/mnist_cnn.pt`：训练好的 CNN 模型。
	- `figures/concept_samples.png`：概念与随机样本展示。
	- `figures/tcav_scores.png`：TCAV 分数对比图。
	- `figures/directional_derivative_hist.png`：方向导数分布图。
	- `report/tcav_report.md`：实验报告。
	- `report/summary.json`：数值汇总。

- `outputs/stroke_concepts/`
	- `figures/stroke_concepts_tcav8.png`：各概念对数字 8 的 TCAV 对比。
	- `figures/stroke_concepts_gap_8_minus_3.png`：数字 8 与数字 3 的差值对比。
	- `report/stroke_concepts_report.md`：多概念实验报告。
	- `report/stroke_concepts_summary.json`：多概念实验汇总。

- `outputs/stroke_concepts_pytorch/`
	- `figures/stroke_concepts_tcav8_pytorch.png`：PyTorch CAV 版本的对比图。
	- `figures/stroke_concepts_gap_8_minus_3_pytorch.png`：PyTorch CAV 版本的差值图。
	- `report/stroke_concepts_report_pytorch.md`：PyTorch CAV 实验报告。
	- `report/stroke_concepts_summary_pytorch.json`：PyTorch CAV 汇总。

## 5) 多概念脚本使用的概念

这两个多概念脚本会依次测试以下 6 个概念：

- `x_shape`
- `vertical`
- `horizontal`
- `slash`
- `backslash`
- `loop`

## 6) 说明

- 多概念脚本都依赖 `run_tcav.py` 先生成的模型检查点。
- 这三个脚本当前都使用固定的输出目录；如果需要改成可配置输出目录，需要再改脚本参数。
- PyTorch CAV 版本会更好利用 GPU；传统线性 CAV 版本仍然保留，便于结果对照。
