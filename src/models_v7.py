"""Version 7 — mở rộng src/models_v4.py: tính ĐỦ bộ metric (accuracy,
precision, recall, f1, g_mean, pr_auc, brier, auc_roc) cho MỖI FOLD trong
5-fold CV, không chỉ AUC-ROC như v4.

File này KHÔNG sửa src/models_v4.py. Lý do bổ sung: v4 chỉ tính AUC-ROC cho
mỗi fold (tiêu chí duy nhất để chọn best_params), nên không biết được bộ tham
số "thắng" theo AUC-ROC có đánh đổi precision/recall/F1 tệ hơn bộ khác hay
không — với bài toán ưu tiên thu hồi nợ, recall trên lớp default quan trọng
không kém AUC-ROC.

VẪN GIỮ tiêu chí chọn best_params là AUC-ROC trung bình qua 5 fold (không đổi
logic chọn của v4) — chỉ bổ sung việc TÍNH và LƯU LẠI toàn bộ metric còn lại,
để có thể kiểm tra đánh đổi, không tự động đổi quyết định chọn tham số.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.config import RANDOM_STATE
from src.evaluate_v2 import evaluate_model
from src.models import _oversample_minority
from src.models_v2 import PARAM_GRIDS, _build_estimator

N_SPLITS = 5

_MODELS_NEED_OVERSAMPLE = {"Neural Network"}
_MODELS_NEED_SAMPLE_WEIGHT = {"GBM"}

# Cac metric duoc tinh cho MOI fold (tai su dung evaluate_v2.evaluate_model,
# bo cot confusion_matrix vi khong the lay trung binh/std truc tiep).
METRIC_KEYS = ["auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1", "g_mean", "brier"]


def _fit_one_fold(model_name: str, params: dict, pos_weight: float, X_tr, y_tr):
    fit_X, fit_y = X_tr, y_tr
    sample_weight = None
    if model_name in _MODELS_NEED_OVERSAMPLE:
        fit_X, fit_y = _oversample_minority(X_tr, y_tr, RANDOM_STATE)
    elif model_name in _MODELS_NEED_SAMPLE_WEIGHT:
        sample_weight = compute_sample_weight("balanced", y_tr)

    model = _build_estimator(model_name, params, pos_weight)
    if sample_weight is not None:
        model.fit(fit_X, fit_y, sample_weight=sample_weight)
    else:
        model.fit(fit_X, fit_y)
    return model


def train_and_select_cv(model_name: str, X_train, y_train, n_splits: int = N_SPLITS):
    """Giống src.models_v4.train_and_select_cv, nhưng mỗi fold tính ĐỦ
    METRIC_KEYS (không chỉ AUC-ROC). Chọn best_params vẫn theo mean AUC-ROC.

    Trả về: (best_model, best_params, best_mean_metrics, all_results)
    - best_mean_metrics: dict {metric: mean_qua_5_fold} cho bộ tham số tốt nhất.
    - all_results: list dict {model, params, mean_<metric>, std_<metric> cho
      từng metric trong METRIC_KEYS} cho MỌI tổ hợp trong grid.
    """
    y_train = np.asarray(y_train)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(X_train, y_train))

    best_mean_auc = -np.inf
    best_params = None
    best_mean_metrics = None
    all_results = []

    for params in PARAM_GRIDS[model_name]:
        fold_metrics = []
        for train_idx, val_idx in fold_indices:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = _fit_one_fold(model_name, params, pos_weight, X_fold_train, y_fold_train)
            metrics = evaluate_model(model, X_fold_val, y_fold_val)
            fold_metrics.append(metrics)

        mean_metrics = {}
        row = {"model": model_name, "params": params}
        for key in METRIC_KEYS:
            values = [m[key] for m in fold_metrics]
            mean_metrics[key] = float(np.mean(values))
            row[f"mean_{key}"] = float(np.mean(values))
            row[f"std_{key}"] = float(np.std(values))
        all_results.append(row)

        if mean_metrics["auc_roc"] > best_mean_auc:
            best_mean_auc = mean_metrics["auc_roc"]
            best_params = params
            best_mean_metrics = mean_metrics

    best_model = _fit_one_fold(model_name, best_params, pos_weight, X_train, y_train)

    return best_model, best_params, best_mean_metrics, all_results


def train_all_models_cv(X_train, y_train, n_splits: int = N_SPLITS, model_names=None) -> tuple[dict, pd.DataFrame]:
    """Chạy train_and_select_cv cho toàn bộ (hoặc một phần) model. Trả về
    (results, all_results_df). results[model_name] chứa "cv_metrics" là dict
    đầy đủ (auc_roc, pr_auc, accuracy, precision, recall, f1, g_mean, brier)
    TRUNG BÌNH qua 5 fold cho bộ tham số tốt nhất — dùng để in ra bảng ở Bước 5
    thay vì chỉ 1 cột AUC-ROC như v4.
    """
    if model_names is None:
        model_names = list(PARAM_GRIDS.keys())

    results = {}
    all_rows = []
    for model_name in model_names:
        model, params, mean_metrics, all_results = train_and_select_cv(
            model_name, X_train, y_train, n_splits=n_splits
        )
        results[model_name] = {
            "model": model,
            "best_params": params,
            "val_auc": mean_metrics["auc_roc"],  # giu ten "val_auc" de tuong thich build_metrics_table
            "cv_metrics": mean_metrics,
        }
        all_rows.extend(all_results)

    return results, pd.DataFrame(all_rows)
