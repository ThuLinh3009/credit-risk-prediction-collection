"""Version 2 — mở rộng src/evaluate.py: thêm PR-AUC, G-mean, Brier score.

File này KHÔNG sửa src/evaluate.py. Các metric mới dùng để:
- G-mean/PR-AUC: so sánh ngang hàng (apples-to-apples) với paper Ampomah et al.
  (2025) (papers/paper 2.pdf), vốn dùng đúng 2 metric này thay vì AUC-ROC đơn
  thuần cho bài toán mất cân bằng.
- Brier score: kiểm tra risk_score còn đại diện xác suất thật hay không sau
  resampling (xem src/resampling_experiment_v2.py) — quan trọng vì
  priority_score (src/priority.py) cần risk_score là xác suất thực, đúng tinh
  thần Sorting Smoothing Method của Yeh & Lien (2009).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from src.evaluate import evaluate_model as evaluate_model_v1


def g_mean_score(y_test, pred) -> float:
    """Trung bình nhân (geometric mean) của recall và specificity — metric
    chính paper Ampomah et al. (2025) dùng để đánh giá dữ liệu mất cân bằng.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(np.sqrt(specificity * recall))


def evaluate_model(model, X_test, y_test) -> dict:
    """Giống src.evaluate.evaluate_model, bổ sung pr_auc, g_mean, brier."""
    metrics = evaluate_model_v1(model, X_test, y_test)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics["pr_auc"] = average_precision_score(y_test, proba)
    metrics["g_mean"] = g_mean_score(y_test, pred)
    # Brier score = MSE giua proba du doan va nhan thuc (0/1); cang thap cang
    # tot. Dung de kiem tra risk_score con dai dien xac suat that hay khong.
    metrics["brier"] = brier_score_loss(y_test, proba)
    return metrics


def build_metrics_table(trained_results: dict, X_test, y_test) -> pd.DataFrame:
    """trained_results: output của models_v2.train_all_models."""
    rows = []
    for model_name, info in trained_results.items():
        metrics = evaluate_model(info["model"], X_test, y_test)
        rows.append(
            {
                "model": model_name,
                "best_params": info["best_params"],
                "val_auc": info["val_auc"],
                "test_auc_roc": metrics["auc_roc"],
                "test_pr_auc": metrics["pr_auc"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_g_mean": metrics["g_mean"],
                "test_brier": metrics["brier"],
            }
        )
    return pd.DataFrame(rows).sort_values("test_auc_roc", ascending=False).reset_index(drop=True)
