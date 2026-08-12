"""Version 2 — mở rộng src/models.py: thêm GBM và LightGBM.

File này KHÔNG sửa src/models.py (pipeline gốc/v1 giữ nguyên 100%). Việc thêm
2 model boosting này tham khảo hướng đi của Ampomah et al. (2025)
(papers/paper 2.pdf) — GBM/LightGBM cùng họ tree-boosting với XGBoost nên
tương thích tốt với cấu trúc pipeline hiện có. Ngược lại, Boruta/DBSCAN/SMOTE
của paper đó KHÔNG được đưa vào đây (xem src/resampling_experiment_v2.py để
biết SMOTE được thử nghiệm riêng, có kiểm soát, vì sao).

Xử lý mất cân bằng, mỗi model một cách phù hợp với API riêng:
- Logistic Regression, Random Forest, LightGBM: class_weight="balanced".
- XGBoost: scale_pos_weight = tỷ lệ lớp 0 / lớp 1.
- GBM (sklearn GradientBoostingClassifier): KHÔNG hỗ trợ class_weight, nên
  dùng sample_weight="balanced" khi fit() để đạt hiệu quả tương đương.
- Neural Network (MLP): oversample thủ công (kế thừa từ src/models.py).
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.config import RANDOM_STATE
from src.models import PARAM_GRIDS as PARAM_GRIDS_V1
from src.models import _build_estimator as _build_estimator_v1
from src.models import _oversample_minority

_NEW_MODEL_GRIDS = {
    "GBM": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in [200, 400]
        for d in [3, 5]
        for lr in [0.05, 0.1]
    ],
    "LightGBM": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in [200, 400]
        for d in [3, 5]
        for lr in [0.05, 0.1]
    ],
}

# PARAM_GRIDS cua v1 (4 model) + 2 model moi (GBM, LightGBM).
PARAM_GRIDS = {**PARAM_GRIDS_V1, **_NEW_MODEL_GRIDS}

_MODELS_NEED_OVERSAMPLE = {"Neural Network"}
_MODELS_NEED_SAMPLE_WEIGHT = {"GBM"}


def _build_estimator(model_name: str, params: dict, pos_weight: float):
    if model_name == "GBM":
        # GradientBoostingClassifier khong co class_weight; mat can bang
        # duoc xu ly bang sample_weight="balanced" truyen qua fit() (xem
        # train_and_select), tuong duong ve hieu qua voi class_weight="balanced".
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    # 4 model con lai: tai su dung nguyen logic cua src/models.py (khong sua doi).
    return _build_estimator_v1(model_name, params, pos_weight)


def train_and_select(model_name: str, X_train, y_train, X_val, y_val):
    """Giống src.models.train_and_select, nhưng dùng PARAM_GRIDS/_build_estimator
    mở rộng (thêm GBM, LightGBM) và truyền sample_weight cho GBM khi fit().
    """
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    fit_X, fit_y = X_train, y_train
    fit_sample_weight = None
    if model_name in _MODELS_NEED_OVERSAMPLE:
        fit_X, fit_y = _oversample_minority(X_train, np.asarray(y_train), RANDOM_STATE)
    elif model_name in _MODELS_NEED_SAMPLE_WEIGHT:
        fit_sample_weight = compute_sample_weight("balanced", y_train)

    best_auc = -np.inf
    best_params = None
    best_model = None

    for params in PARAM_GRIDS[model_name]:
        model = _build_estimator(model_name, params, pos_weight)
        if fit_sample_weight is not None:
            model.fit(fit_X, fit_y, sample_weight=fit_sample_weight)
        else:
            model.fit(fit_X, fit_y)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        if val_auc > best_auc:
            best_auc, best_params, best_model = val_auc, params, model

    return best_model, best_params, best_auc


def train_all_models(X_train, y_train, X_val, y_val) -> dict:
    """Huấn luyện & tinh chỉnh toàn bộ 6 mô hình (4 gốc + GBM + LightGBM)."""
    results = {}
    for model_name in PARAM_GRIDS:
        model, params, val_auc = train_and_select(model_name, X_train, y_train, X_val, y_val)
        results[model_name] = {
            "model": model,
            "best_params": params,
            "val_auc": val_auc,
        }
    return results
