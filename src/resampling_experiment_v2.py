"""Version 2 — Thí nghiệm so sánh riêng: SMOTE vs class_weight.

QUAN TRỌNG: đây là một thí nghiệm SO SÁNH, KHÔNG thay thế pipeline chính
(src/models.py + src/models_v2.py, dùng class_weight/sample_weight/oversample
thủ công). File này KHÔNG sửa bất kỳ file gốc nào. Lý do tách riêng thay vì áp
dụng thẳng SMOTE của Ampomah et al. (2025) (papers/paper 2.pdf) vào pipeline
chính:

1. Công thức priority_score = risk_score x mức_độ_trễ_hạn (src/priority.py)
   cần risk_score LÀ MỘT XÁC SUẤT THỰC (0-1), đúng tinh thần Sorting Smoothing
   Method của Yeh & Lien (2009) (papers/paper 1.pdf) mà dự án đang đối chiếu.
   Khi train trên dữ liệu đã SMOTE cân bằng 50/50 (khác hẳn tỷ lệ default thật
   ~22%), xác suất dự đoán của model bị lệch (over-estimate risk) — nên không
   thể dùng thẳng risk_score sau SMOTE để tính priority_score nếu chưa hiệu
   chỉnh lại (calibrate).
2. Paper 2 đánh giá bằng F1/G-mean (đo hiệu năng phân loại), không đo xem
   xác suất dự đoán có "thật" hay không — nên hiệu năng phân loại tốt hơn sau
   SMOTE không đồng nghĩa risk_score đáng tin hơn cho bài toán xếp hạng ưu
   tiên thu hồi nợ.

Thiết kế thí nghiệm:
- SMOTE chỉ áp dụng trên TẬP TRAIN (không đụng vào validation/test, để giữ
  đúng phân phối thực ~22% default khi đánh giá — tránh đánh giá lạc quan
  giả tạo do rò rỉ phân phối nhân tạo sang tập test).
- Vì tập train sau SMOTE đã cân bằng 50/50, các model KHÔNG dùng thêm
  class_weight/scale_pos_weight nữa (tránh "cân bằng kép").
- So sánh 3 phiên bản cho mỗi model: (1) class_weight (baseline hiện tại),
  (2) SMOTE chưa hiệu chỉnh, (3) SMOTE + hiệu chỉnh lại xác suất
  (CalibratedClassifierCV trên tập validation gốc). Cột `brier` trong bảng so
  sánh cho biết risk_score có còn đáng tin sau SMOTE hay không.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.evaluate_v2 import evaluate_model
from src.models_v2 import PARAM_GRIDS

# Model duoc thu nghiem lai voi SMOTE. Bo qua Neural Network (da co co che
# oversample thu cong rieng trong pipeline chinh, khong can thu nghiem SMOTE
# lai) va giu nguyen danh sach con lai de so sanh cong bang voi baseline.
SMOTE_MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "GBM", "LightGBM"]


def _build_estimator_unweighted(model_name: str, params: dict):
    """Bản KHÔNG dùng class_weight/scale_pos_weight — vì tập train sau SMOTE
    đã cân bằng 50/50, cộng thêm class_weight sẽ gây "cân bằng kép" (over-
    correct), khiến model quá thiên vị lớp default.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(C=params["C"], max_iter=2000, random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "GBM":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    raise ValueError(f"Model khong duoc ho tro trong thi nghiem SMOTE: {model_name}")


def run_smote_experiment(X_train, y_train, X_val, y_val, model_names: list[str] | None = None) -> dict:
    """SMOTE chỉ trên tập train; chọn tham số tốt nhất theo AUC-ROC trên tập
    validation GỐC (không SMOTE) — giữ validation phản ánh đúng phân phối
    thực ~22% default, tránh đánh giá lạc quan giả tạo.
    """
    if model_names is None:
        model_names = SMOTE_MODEL_NAMES

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"  SMOTE: train {len(y_train)} -> {len(y_train_res)} dong "
        f"(default: {int(np.asarray(y_train).sum())} -> {int(np.asarray(y_train_res).sum())})"
    )

    results = {}
    for model_name in model_names:
        best_auc, best_params, best_model = -np.inf, None, None
        for params in PARAM_GRIDS[model_name]:
            model = _build_estimator_unweighted(model_name, params)
            model.fit(X_train_res, y_train_res)
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            if val_auc > best_auc:
                best_auc, best_params, best_model = val_auc, params, model
        results[model_name] = {"model": best_model, "best_params": best_params, "val_auc": best_auc}

    return results


def calibrate_model(model, X_val, y_val, method: str = "sigmoid"):
    """Hiệu chỉnh lại xác suất dự đoán của model đã train trên dữ liệu SMOTE
    (bị lệch vì train trên phân phối 50/50 nhân tạo), dùng tập validation GỐC
    (phân phối thật ~22% default) để đưa risk_score về gần xác suất thực hơn
    trước khi có thể dùng cho công thức priority_score.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_val, y_val)
    return calibrated


def build_resampling_comparison_table(
    baseline_results: dict,
    smote_results: dict,
    X_test,
    y_test,
    X_val,
    y_val,
) -> pd.DataFrame:
    """So sánh 3 phiên bản cho từng model: (1) class_weight (baseline pipeline
    chính), (2) SMOTE chưa hiệu chỉnh, (3) SMOTE + hiệu chỉnh lại xác suất.
    Cột `brier` (càng thấp càng tốt) cho biết risk_score còn đáng tin hay
    không sau SMOTE — đây là điểm mấu chốt để quyết định có dùng model SMOTE
    cho priority_score hay không, chứ không chỉ nhìn AUC-ROC/F1/G-mean.
    """
    rows = []
    for model_name in smote_results:
        if model_name not in baseline_results:
            continue

        base_metrics = evaluate_model(baseline_results[model_name]["model"], X_test, y_test)
        rows.append({"model": model_name, "pipeline": "class_weight (baseline)", **_flatten(base_metrics)})

        smote_model = smote_results[model_name]["model"]
        smote_metrics = evaluate_model(smote_model, X_test, y_test)
        rows.append({"model": model_name, "pipeline": "SMOTE (chua hieu chinh)", **_flatten(smote_metrics)})

        calibrated = calibrate_model(smote_model, X_val, y_val)
        calibrated_metrics = evaluate_model(calibrated, X_test, y_test)
        rows.append({"model": model_name, "pipeline": "SMOTE + hieu chinh (calibrated)", **_flatten(calibrated_metrics)})

    cols = ["model", "pipeline", "auc_roc", "pr_auc", "f1", "g_mean", "recall", "precision", "brier"]
    return pd.DataFrame(rows)[cols]


def _flatten(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "confusion_matrix"}
