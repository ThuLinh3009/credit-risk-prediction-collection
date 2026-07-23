"""Sprint 3 — Diễn giải mô hình bằng SHAP / feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_values(model, X_background: np.ndarray, feature_names: list[str], sample_size: int = 500):
    """Tính SHAP values cho một mẫu con của tập dữ liệu (để giới hạn thời
    gian chạy). Dùng TreeExplainer cho mô hình cây, Explainer chung cho
    các mô hình còn lại.
    """
    import shap

    rng = np.random.RandomState(42)
    n = min(sample_size, X_background.shape[0])
    idx = rng.choice(X_background.shape[0], size=n, replace=False)
    X_sample = X_background[idx]

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.Explainer(model.predict_proba, X_sample)
        sv = explainer(X_sample)
        shap_values = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

    return shap_values, X_sample


def top_features_by_shap(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
