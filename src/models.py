"""Sprint 2 — Model Building, Training & Tuning.

4 mô hình so sánh: Logistic Regression, Random Forest, XGBoost,
Neural Network (MLP). Xử lý mất cân bằng bằng class-weight (LR, RF,
XGBoost) hoặc oversampling thủ công (MLP, vì MLPClassifier của
scikit-learn không hỗ trợ tham số class_weight/sample_weight).
Tinh chỉnh siêu tham số bằng tìm kiếm lưới nhỏ (grid search thủ công),
chọn mô hình theo AUC-ROC trên tập validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

PARAM_GRIDS = {
    "Logistic Regression": [
        {"C": c} for c in [0.01, 0.1, 1.0, 10.0]
    ],
    "Random Forest": [
        {"n_estimators": n, "max_depth": d}
        for n in [200, 400]
        for d in [6, 10, None]
    ],
    "XGBoost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in [200, 400]
        for d in [3, 5]
        for lr in [0.05, 0.1]
    ],
    "Neural Network": [
        {"hidden_layer_sizes": hl, "alpha": a}
        for hl in [(32,), (64, 32)]
        for a in [1e-4, 1e-3]
    ],
}


def _build_estimator(model_name: str, params: dict, pos_weight: float):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "Neural Network":
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _oversample_minority(X: np.ndarray, y: np.ndarray, random_state: int):
    """MLPClassifier không hỗ trợ class_weight, nên cân bằng lớp bằng
    cách lặp lại (oversample) các quan sát default=1 cho bằng số lượng
    default=0 trong tập huấn luyện.
    """
    idx_majority = np.where(y == 0)[0]
    idx_minority = np.where(y == 1)[0]
    idx_minority_upsampled = resample(
        idx_minority,
        replace=True,
        n_samples=len(idx_majority),
        random_state=random_state,
    )
    idx = np.concatenate([idx_majority, idx_minority_upsampled])
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)
    return X[idx], y[idx]


def train_and_select(model_name: str, X_train, y_train, X_val, y_val):
    """Chạy grid search thủ công cho một mô hình, chọn bộ tham số
    tốt nhất theo AUC-ROC trên tập validation.

    Trả về: (best_estimator_đã_fit_trên_train, best_params, best_val_auc)
    """
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    fit_X, fit_y = X_train, y_train
    if model_name == "Neural Network":
        fit_X, fit_y = _oversample_minority(X_train, np.asarray(y_train), RANDOM_STATE)

    best_auc = -np.inf
    best_params = None
    best_model = None

    for params in PARAM_GRIDS[model_name]:
        model = _build_estimator(model_name, params, pos_weight)
        model.fit(fit_X, fit_y)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        if val_auc > best_auc:
            best_auc, best_params, best_model = val_auc, params, model

    return best_model, best_params, best_auc


def train_all_models(X_train, y_train, X_val, y_val) -> dict:
    """Huấn luyện & tinh chỉnh cả 4 mô hình, trả về dict kết quả."""
    results = {}
    for model_name in PARAM_GRIDS:
        model, params, val_auc = train_and_select(model_name, X_train, y_train, X_val, y_val)
        results[model_name] = {
            "model": model,
            "best_params": params,
            "val_auc": val_auc,
        }
    return results
