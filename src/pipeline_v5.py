"""Version 5 — pipeline CRISP-DM dùng chia 80/20 TRỰC TIẾP (thay vì 60/20/20
rồi gộp train+val như version 4), giữ nguyên cách tune (grid + 5-fold CV,
src/models_v4.py, không đổi).

File này KHÔNG sửa src/pipeline.py/_v2/_v3/_v4. Chỉ đổi src/data.py:split()
(60/20/20) thành src/data_v5.py:split_80_20() (80/20 trực tiếp).

QUAN TRỌNG: vì cách chia khác thuật toán (1-bước vs 2-bước), tập test của
version 5 KHÁC tập test của version 1-4 (~50% ID trùng nhau, xem
notebooks/05_credit_risk_pipeline_v5.ipynb để kiểm chứng bằng số liệu) — nên
so sánh test_auc_roc giữa v5 và các version khác chỉ mang tính tham khảo,
không phải so sánh trên cùng 1 tập dữ liệu.

Kết quả lưu vào reports/ với tiền tố "v5_".

Chạy: py -m src.pipeline_v5
"""

from __future__ import annotations

import json

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data import clean, load_raw
from src.data_v5 import split_80_20
from src.evaluate import build_paper_comparison_table
from src.evaluate_v2 import build_metrics_table
from src.features import add_derived_features, build_preprocessor
from src.models_v4 import train_all_models_cv
from src.priority import build_priority_ranking


def main(n_splits: int = 5):
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("[1/6] Nap va lam sach du lieu...")
    df = clean(load_raw())
    df_fe = add_derived_features(df)

    print("[2/6] Chia 80/20 TRUC TIEP (khac 60/20/20 cua v1-v4)...")
    X_train, X_test, y_train, y_test = split_80_20(df_fe)
    print(f"  train={len(X_train)} test={len(X_test)}")

    print("[3/6] Tien xu ly (one-hot + scale)...")
    feature_cols = list(X_train.columns)
    preprocessor = build_preprocessor(feature_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    print(f"[4/6] Grid search + {n_splits}-fold CV cho 6 mo hinh (dung PARAM_GRIDS cua models_v2)...")
    trained, all_results_df = train_all_models_cv(X_train_t, y_train.to_numpy(), n_splits=n_splits)
    all_results_df.to_csv(REPORTS_DIR / "v5_cv_all_results.csv", index=False)
    for name, info in trained.items():
        print(
            f"  {name}: best_params={info['best_params']} "
            f"mean_cv_auc={info['val_auc']:.4f} std={info['cv_std_auc']:.4f}"
        )

    print("[5/6] Danh gia tren tap test & so sanh voi paper goc Yeh & Lien (2009)...")
    metrics_table = build_metrics_table(trained, X_test_t, y_test.to_numpy())
    comparison_table = build_paper_comparison_table(metrics_table)

    metrics_table.to_csv(REPORTS_DIR / "v5_model_metrics.csv", index=False)
    comparison_table.to_csv(REPORTS_DIR / "v5_comparison_with_paper_2009.csv", index=False)
    print(metrics_table.to_string(index=False))
    print(comparison_table.to_string(index=False))

    best_model_name = metrics_table.iloc[0]["model"]
    best_model = trained[best_model_name]["model"]
    print(f"  => Mo hinh tot nhat theo AUC-ROC tren test: {best_model_name}")

    print("[6/6] SHAP feature importance & bang xep hang uu tien thu hoi no...")
    try:
        from src.interpret import compute_shap_values, top_features_by_shap

        shap_values, X_sample = compute_shap_values(best_model, X_test_t, feature_names)
        top_features = top_features_by_shap(shap_values, feature_names)
        top_features.to_csv(REPORTS_DIR / "v5_shap_top_features.csv", index=False)
        print(top_features.to_string(index=False))
    except ImportError:
        print("  (bo qua SHAP: chua cai dat thu vien 'shap')")

    risk_scores = best_model.predict_proba(X_test_t)[:, 1]
    priority_ranking = build_priority_ranking(X_test, X_test.index.to_series(name="ID"), risk_scores)
    priority_ranking.to_csv(REPORTS_DIR / "v5_priority_ranking.csv", index=False)
    print(priority_ranking.head(10).to_string(index=False))

    summary = {
        "best_model": best_model_name,
        "best_model_test_auc": float(metrics_table.iloc[0]["test_auc_roc"]),
        "n_splits_cv": n_splits,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "split_method": "80/20 truc tiep (1 buoc), KHAC voi 60/20/20 cua v1-v4",
    }
    with open(REPORTS_DIR / "v5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nHoan tat (version 5, chia 80/20 truc tiep). Ket qua da luu trong thu muc reports/ (tien to 'v5_').")


if __name__ == "__main__":
    main()
