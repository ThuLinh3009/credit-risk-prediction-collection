"""Sprint 1 — Data Understanding & Preparation.

Nạp dữ liệu, làm sạch giá trị bất thường, và chia tập
train/validation/test theo tỷ lệ stratify.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    ID_COL,
    PAY_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Nạp dữ liệu gốc UCI Default of Credit Card Clients."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý giá trị bất thường theo Sprint 1 của đề xuất dự án.

    - Đổi tên PAY_0 -> PAY_1 để nhất quán với PAY_2..PAY_6.
    - EDUCATION: 0, 5, 6 không nằm trong định nghĩa gốc (1=graduate,
      2=university, 3=high school, 4=others) -> gộp vào nhóm 4 (others).
    - MARRIAGE: 0 không nằm trong định nghĩa gốc (1=married, 2=single,
      3=others) -> gộp vào nhóm 3 (others).
    """
    df = df.rename(columns={"PAY_0": "PAY_1"})

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def split(df: pd.DataFrame):
    """Chia stratified train/validation/test theo TRAIN_SIZE/VAL_SIZE/TEST_SIZE.

    ID_COL được đặt làm index của X_train/X_val/X_test để giữ lại liên kết
    với khách hàng gốc (dùng cho bảng xếp hạng ưu tiên thu hồi nợ).
    """
    assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-9

    df = df.set_index(ID_COL)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    val_ratio_of_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio_of_temp,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare():
    """Tiện ích: nạp, làm sạch và chia dữ liệu trong một bước."""
    df = clean(load_raw())
    return split(df)
