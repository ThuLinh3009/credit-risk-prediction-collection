"""Đường dẫn và hằng số dùng chung cho pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "datasets" / "UCI_Credit_Card.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COL = "default.payment.next.month"
ID_COL = "ID"

PAY_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RANDOM_STATE = 42

# Tỷ lệ chia train/validation/test (stratify theo target)
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# Kết quả gốc trích từ Table 1 & Table 2, Yeh & Lien (2009),
# Expert Systems with Applications 36, 2473-2480 (papers/5398933.pdf).
# Area ratio trong lift chart tương đương định nghĩa hệ số Gini
# chuẩn hoá (2*AUC - 1), nên AUC_approx = (area_ratio + 1) / 2.
# Đây là quy đổi gần đúng để đối chiếu, không phải số liệu AUC-ROC
# được tác giả gốc tính trực tiếp (paper dùng lift chart, không dùng AUC-ROC).
PAPER_2009_RESULTS = {
    "K-Nearest Neighbor": {"error_rate_val": 0.16, "area_ratio_val": 0.45},
    "Logistic Regression": {"error_rate_val": 0.18, "area_ratio_val": 0.44},
    "Discriminant Analysis": {"error_rate_val": 0.26, "area_ratio_val": 0.43},
    "Naive Bayesian": {"error_rate_val": 0.21, "area_ratio_val": 0.53},
    "Neural Network": {"error_rate_val": 0.17, "area_ratio_val": 0.54},
    "Classification Tree": {"error_rate_val": 0.17, "area_ratio_val": 0.536},
}

for _name, _res in PAPER_2009_RESULTS.items():
    _res["auc_approx"] = (_res["area_ratio_val"] + 1) / 2
    _res["accuracy_val"] = 1 - _res["error_rate_val"]
