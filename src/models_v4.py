"""Version 4 — grid search + 5-fold cross-validation (thay 1 validation set
cố định của v1/v2 bằng trung bình 5-fold StratifiedKFold trên train).

File này KHÔNG sửa src/models.py, src/models_v2.py, src/models_v3.py. Dùng
LẠI ĐÚNG PARAM_GRIDS và cách dựng estimator của src/models_v2.py (không đổi
lưới tham số) — mục đích là cô lập đúng 1 biến số cần so sánh: "đánh giá bằng
1 validation set" (v1/v2) vs "đánh giá bằng trung bình 5-fold CV" (v4), giữ
nguyên mọi thứ khác. Đây cũng là cách Ampomah et al. (2025) (papers/paper
2.pdf, Section 3.6) làm cho GBM ("grid search with 5-fold cross-validation").

Vì sao không dùng thẳng sklearn.model_selection.GridSearchCV:
- Neural Network cần oversample thủ công lớp thiểu số CHỈ trên phần train của
  từng fold (không phải oversample rồi mới chia fold — sẽ làm rò rỉ mẫu trùng
  lặp giữa fold train/val). GridSearchCV không tự làm việc này; cần vòng lặp
  CV thủ công để kiểm soát đúng thứ tự oversample-trong-từng-fold.
- GBM cần truyền sample_weight="balanced" khi fit() (không phải tham số
  constructor) — GridSearchCV hỗ trợ qua fit_params nhưng dùng vòng lặp thủ
  công ở đây cho nhất quán với cách models.py/models_v2.py đã viết.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.config import RANDOM_STATE
from src.models import _oversample_minority
from src.models_v2 import PARAM_GRIDS, _build_estimator

N_SPLITS = 5

_MODELS_NEED_OVERSAMPLE = {"Neural Network"}
_MODELS_NEED_SAMPLE_WEIGHT = {"GBM"}


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
    """Grid search với n_splits-fold StratifiedKFold CV trên TRAIN. Chọn bộ
    tham số theo AUC-ROC TRUNG BÌNH qua các fold (thay vì 1 validation set cố
    định). Refit bộ tham số tốt nhất trên TOÀN BỘ train để lấy model cuối.

    Trả về: (best_model, best_params, best_mean_auc, best_std_auc, all_results)
    all_results: list các dict {params, mean_cv_auc, std_cv_auc, fold_aucs} cho
    MỌI tổ hợp trong grid — dùng để kiểm tra độ ổn định (so sánh với v1/v2/v3).
    """
    y_train = np.asarray(y_train)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(X_train, y_train))

    best_mean_auc, best_std_auc, best_params = -np.inf, None, None
    all_results = []

    for params in PARAM_GRIDS[model_name]:
        fold_aucs = []
        for train_idx, val_idx in fold_indices:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = _fit_one_fold(model_name, params, pos_weight, X_fold_train, y_fold_train)
            fold_auc = roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
            fold_aucs.append(fold_auc)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        all_results.append(
            {
                "model": model_name,
                "params": params,
                "mean_cv_auc": mean_auc,
                "std_cv_auc": std_auc,
                "fold_aucs": fold_aucs,
            }
        )
        if mean_auc > best_mean_auc:
            best_mean_auc, best_std_auc, best_params = mean_auc, std_auc, params

    # Refit tren toan bo train (khong chia fold) voi bo tham so tot nhat.
    best_model = _fit_one_fold(model_name, best_params, pos_weight, X_train, y_train)

    return best_model, best_params, best_mean_auc, best_std_auc, all_results


def train_all_models_cv(X_train, y_train, n_splits: int = N_SPLITS, model_names=None) -> tuple[dict, pd.DataFrame]:
    """Chạy train_and_select_cv cho toàn bộ (hoặc một phần) model. Trả về
    (results, all_results_df) — all_results_df gộp kết quả MỌI tổ hợp tham số
    của MỌI model (không chỉ best), tương đương trials_df của Optuna (v3).
    """
    if model_names is None:
        model_names = list(PARAM_GRIDS.keys())

    results = {}
    all_rows = []
    for model_name in model_names:
        model, params, mean_auc, std_auc, all_results = train_and_select_cv(
            model_name, X_train, y_train, n_splits=n_splits
        )
        results[model_name] = {
            "model": model,
            "best_params": params,
            "val_auc": mean_auc,  # ten "val_auc" giu nguyen de tuong thich voi build_metrics_table
            "cv_std_auc": std_auc,
        }
        for row in all_results:
            all_rows.append(
                {
                    "model": row["model"],
                    "params": row["params"],
                    "mean_cv_auc": row["mean_cv_auc"],
                    "std_cv_auc": row["std_cv_auc"],
                    **{f"fold_{i+1}_auc": v for i, v in enumerate(row["fold_aucs"])},
                }
            )

    return results, pd.DataFrame(all_rows)
