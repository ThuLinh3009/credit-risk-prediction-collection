"""Sprint 3 — Evaluation & so sánh với paper gốc Yeh & Lien (2009)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PAPER_2009_RESULTS


def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred),
    }


def build_metrics_table(trained_results: dict, X_test, y_test) -> pd.DataFrame:
    """trained_results: output của models.train_all_models (dict model_name -> info)."""
    rows = []
    for model_name, info in trained_results.items():
        metrics = evaluate_model(info["model"], X_test, y_test)
        rows.append(
            {
                "model": model_name,
                "best_params": info["best_params"],
                "val_auc": info["val_auc"],
                "test_auc_roc": metrics["auc_roc"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("test_auc_roc", ascending=False).reset_index(drop=True)


def build_paper_comparison_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """So sánh AUC-ROC hiện tại với AUC quy đổi gần đúng từ area ratio
    trong paper gốc (xem ghi chú quy đổi ở src/config.py).

    Chỉ 2 mô hình có tương ứng trực tiếp trong paper gốc:
    Logistic Regression và Neural Network. Random Forest / XGBoost
    không xuất hiện trong paper 2009 (kỹ thuật ensemble hiện đại hơn)
    nên được đánh dấu "khong co trong paper goc".
    """
    name_map = {
        "Logistic Regression": "Logistic Regression",
        "Neural Network": "Neural Network",
    }

    rows = []
    for _, row in metrics_table.iterrows():
        paper_key = name_map.get(row["model"])
        if paper_key and paper_key in PAPER_2009_RESULTS:
            paper = PAPER_2009_RESULTS[paper_key]
            paper_auc = paper["auc_approx"]
            improvement_pct = (row["test_auc_roc"] - paper_auc) / paper_auc * 100
            rows.append(
                {
                    "model": row["model"],
                    "auc_2026": row["test_auc_roc"],
                    "auc_approx_2009": paper_auc,
                    "improvement_pct": improvement_pct,
                    "note": "quy doi tu area ratio (Table 1, Yeh & Lien 2009)",
                }
            )
        else:
            rows.append(
                {
                    "model": row["model"],
                    "auc_2026": row["test_auc_roc"],
                    "auc_approx_2009": np.nan,
                    "improvement_pct": np.nan,
                    "note": "khong co trong paper goc (2009)",
                }
            )
    return pd.DataFrame(rows)
