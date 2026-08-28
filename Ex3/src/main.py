from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import CV_FOLDS, RANDOM_STATE, SCORING, TEST_SIZE
from src.data_utils import load_titanic_dataset, split_features_target
from src.modeling import build_model_pipelines, get_param_grids
from src.reporting import (
    ensure_output_dirs,
    save_best_params,
    save_confusion_matrices,
    save_metrics,
    save_prediction_details,
    save_roc_comparison,
    write_markdown_report,
)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """计算分类任务常用指标。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_compare(verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """训练并比较逻辑回归与随机森林。"""
    ensure_output_dirs()

    df = load_titanic_dataset()
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_model_pipelines()
    param_grids = get_param_grids()

    metrics_rows = []
    best_params = {}
    confusion_payload = {}
    roc_payload = {}
    test_detail_frames = []

    for model_name, pipeline in pipelines.items():
        # 每个模型单独做CV调参，保持公平对比
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring=SCORING,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        pred = best_model.predict(x_test)
        proba = best_model.predict_proba(x_test)[:, 1]

        metrics = evaluate_model(y_test, pred, proba)
        metrics_rows.append({"model": model_name, **metrics})
        best_params[model_name] = search.best_params_

        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, proba)
        confusion_payload[model_name] = {"cm": cm}
        roc_payload[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": metrics["roc_auc"],
        }

        detail = pd.DataFrame(
            {
                "model": model_name,
                "y_true": y_test.values,
                "y_pred": pred,
                "y_proba": proba,
            }
        )
        test_detail_frames.append(detail)

        if verbose:
            print(f"[{model_name}] CV 最优参数: {search.best_params_}")
            print(
                f"[{model_name}] 测试集: "
                f"Acc={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1", ascending=False)

    save_metrics(metrics_df)
    save_best_params(best_params)
    save_prediction_details(pd.concat(test_detail_frames, ignore_index=True))
    save_confusion_matrices(confusion_payload)
    save_roc_comparison(roc_payload)

    lr = metrics_df.loc[metrics_df["model"] == "logistic_regression"].iloc[0]
    rf = metrics_df.loc[metrics_df["model"] == "random_forest"].iloc[0]
    comparison_text = (
        f"随机森林相对逻辑回归在测试集上的准确率变化为 "
        f"{(rf['accuracy'] - lr['accuracy']) * 100:+.2f} 个百分点，"
        f"F1 变化为 {(rf['f1'] - lr['f1']) * 100:+.2f} 个百分点。"
        f"从本次划分看，随机森林{'更好' if rf['f1'] >= lr['f1'] else '不占优'}。"
    )

    report_path = write_markdown_report(metrics_df, comparison_text)

    if verbose:
        print("\n=== 总结 ===")
        print(comparison_text)
        print(f"报告已生成: {report_path}")

    return metrics_df, best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanic: 随机森林与逻辑回归对比实验")
    parser.add_argument("--quiet", action="store_true", help="仅输出必要信息")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_compare(verbose=not args.quiet)


if __name__ == "__main__":
    main()

