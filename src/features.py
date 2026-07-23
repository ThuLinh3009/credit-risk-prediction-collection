"""Sprint 1 — Feature engineering.

Thêm đặc trưng dẫn xuất để giảm đa cộng tuyến giữa BILL_AMT1-6,
mã hoá biến phân loại và chuẩn hoá biến số.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BILL_AMT_COLS, PAY_AMT_COLS, PAY_COLS

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng tổng hợp thay thế một phần BILL_AMT1-6 thô,
    nhằm giảm đa cộng tuyến khi dùng cho Logistic Regression.
    """
    df = df.copy()

    df["AVG_BILL_AMT"] = df[BILL_AMT_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)
    # Xu hướng dư nợ: âm nghĩa là dư nợ đang giảm dần theo thời gian gần đây.
    df["BILL_AMT_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["MAX_DELAY"] = df[PAY_COLS].max(axis=1)
    df["AVG_DELAY"] = df[PAY_COLS].mean(axis=1)
    # Tỷ lệ trả nợ / dư nợ trung bình, tránh chia cho 0.
    df["PAY_TO_BILL_RATIO"] = df["AVG_PAY_AMT"] / df["AVG_BILL_AMT"].replace(0, np.nan)
    df["PAY_TO_BILL_RATIO"] = df["PAY_TO_BILL_RATIO"].fillna(0).clip(-10, 10)

    return df


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tính Variance Inflation Factor để kiểm tra đa cộng tuyến."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[cols].astype(float).values
    vif = pd.DataFrame(
        {
            "feature": cols,
            "VIF": [variance_inflation_factor(X, i) for i in range(len(cols))],
        }
    ).sort_values("VIF", ascending=False)
    return vif


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer dùng chung: one-hot cho biến phân loại,
    chuẩn hoá (StandardScaler) cho biến số. Việc scale không ảnh hưởng
    đến các mô hình cây (RF, XGBoost) nên có thể dùng chung một pipeline
    tiền xử lý cho cả 4 mô hình.
    """
    categorical = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        sparse_threshold=0,  # ep dense output de tuong thich SHAP & cac buoc sau
    )
