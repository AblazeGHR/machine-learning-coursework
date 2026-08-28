from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import FIGURE_DIR, OUTPUT_DIR, REPORT_DIR


def ensure_output_dirs() -> None:
    """创建实验输出目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics_df: pd.DataFrame) -> Path:
    """保存模型指标到CSV。"""
    path = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_best_params(best_params: Dict[str, Dict[str, object]]) -> Path:
    """保存最优参数到JSON。"""
    path = OUTPUT_DIR / "best_params.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    return path


def save_prediction_details(pred_df: pd.DataFrame) -> Path:
    """保存测试集逐样本预测详情。"""
    path = OUTPUT_DIR / "predictions_test.csv"
    pred_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_confusion_matrices(confusion_payload: Dict[str, Dict[str, np.ndarray]]) -> None:
    """保存每个模型的混淆矩阵图。"""
    for model_name, payload in confusion_payload.items():
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=payload["cm"],
            display_labels=["未生还", "生还"],
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{model_name} 混淆矩阵")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"confusion_{model_name}.png", dpi=180)
        plt.close(fig)


def save_roc_comparison(roc_payload: Dict[str, Dict[str, np.ndarray]]) -> Path:
    """保存模型ROC对比图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, payload in roc_payload.items():
        RocCurveDisplay(
            fpr=payload["fpr"],
            tpr=payload["tpr"],
            roc_auc=payload["roc_auc"],
            estimator_name=model_name,
        ).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC 曲线对比")
    fig.tight_layout()

    path = FIGURE_DIR / "roc_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown_report(metrics_df: pd.DataFrame, comparison_text: str) -> Path:
    """输出中文实验报告。"""
    report_path = REPORT_DIR / "实验报告.md"

    pretty = metrics_df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:.4f}")

    table_md = pretty.to_markdown(index=False)

    report = f"""# Titanic 生还预测：随机森林 vs 逻辑回归

## 1. 实验设置
- 数据集：Titanic（优先读取本地 `data/raw/train.csv`，否则使用 OpenML 版本并缓存）
- 划分方式：训练集/测试集 = 80/20，分层抽样
- 调参方式：5 折交叉验证，评分指标为 F1
- 预处理：数值特征中位数填补+标准化；类别特征众数填补+独热编码

## 2. 测试集结果
{table_md}

## 3. 结果解读
{comparison_text}

## 4. 输出文件
- `outputs/metrics.csv`
- `outputs/best_params.json`
- `outputs/predictions_test.csv`
- `outputs/figures/confusion_logistic_regression.png`
- `outputs/figures/confusion_random_forest.png`
- `outputs/figures/roc_comparison.png`
"""

    report_path.write_text(report, encoding="utf-8")
    return report_path

