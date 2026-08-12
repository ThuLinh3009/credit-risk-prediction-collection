"""Version 5 — chia dữ liệu TRỰC TIẾP 80/20 (train/test), không qua bước
60/20/20 rồi gộp lại như version 4.

File này KHÔNG sửa src/data.py. Mục đích: kiểm chứng xem chia trực tiếp 80/20
(1 lần gọi train_test_split) có cho ra tập test KHÁC — và do đó kết quả
tuning/đánh giá có khác — so với cách làm của version 4 (chia 60/20/20 rồi gộp
train+val thành pool 80% cho CV, giữ nguyên tập test 20% ban đầu) hay không.

LƯU Ý QUAN TRỌNG: vì thuật toán chia 1-bước (80/20 trực tiếp) và chia 2-bước
(60/40 rồi 50/50 của 40%) tạo ra 2 chuỗi xáo trộn khác nhau dù cùng
random_state, TẬP TEST CỦA VERSION 5 SẼ KHÔNG TRÙNG với tập test của
V1/V2/V3/V4 — đây chính xác là điều cần kiểm chứng bằng số liệu (xem notebook
05, cell kiểm tra overlap ID), không chỉ suy đoán.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COL, RANDOM_STATE, TARGET_COL


def split_80_20(df: pd.DataFrame):
    """Chia stratified 80/20 TRỰC TIẾP (1 lần gọi train_test_split), khác với
    src/data.py:split() (chia 60/20/20 qua 2 bước train_test_split lồng nhau).

    Trả về X_train, X_test, y_train, y_test — KHÔNG có tập validation riêng,
    vì 5-fold CV (src/models_v4.py:train_all_models_cv) tự tạo validation nội
    bộ từ 80% train, không cần tách thêm.
    """
    df = df.set_index(ID_COL)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test
