"""Version 2 — pipeline CRISP-DM mở rộng: 6 model (thêm GBM, LightGBM) +
thí nghiệm so sánh riêng SMOTE vs class_weight.

File này KHÔNG sửa src/pipeline.py gốc — chạy độc lập, đọc cùng dữ liệu và
cùng bước data/feature (không đổi), nhưng dùng src/models_v2.py,
src/evaluate_v2.py, src/resampling_experiment_v2.py cho phần model.

Kết quả được lưu vào reports/ với tiền tố "v2_" để không ghi đè báo cáo của
pipeline gốc (reports/model_metrics.csv, v.v.).

Chạy: py -m src.pipeline_v2
"""

from __future__ import annotations

import json

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data import clean, load_raw, split
from src.evaluate import build_paper_comparison_table
from src.evaluate_v2 import build_metrics_table
from src.features import add_derived_features, build_preprocessor
from src.models_v2 import train_all_models
from src.priority import build_priority_ranking
from src.resampling_experiment_v2 import build_resampling_comparison_table, run_smote_experiment


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("[1/7] Nap va lam sach du lieu...")
    df = clean(load_raw())
    df_fe = add_derived_features(df)

    print("[2/7] Chia train/validation/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split(df_fe)
    print(f"  train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    print("[3/7] Tien xu ly (one-hot + scale)...")
    feature_cols = list(X_train.columns)
    preprocessor = build_preprocessor(feature_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    print("[4/7] Huan luyen & tinh chinh 6 mo hinh (Logistic Regression, "
          "Random Forest, XGBoost, Neural Network, GBM, LightGBM)...")
    trained = train_all_models(X_train_t, y_train.to_numpy(), X_val_t, y_val.to_numpy())
    for name, info in trained.items():
        print(f"  {name}: best_params={info['best_params']} val_auc={info['val_auc']:.4f}")

    print("[5/7] Danh gia tren tap test & so sanh voi paper goc Yeh & Lien (2009)...")
    metrics_table = build_metrics_table(trained, X_test_t, y_test.to_numpy())
    comparison_table = build_paper_comparison_table(metrics_table)

    metrics_table.to_csv(REPORTS_DIR / "v2_model_metrics.csv", index=False)
    comparison_table.to_csv(REPORTS_DIR / "v2_comparison_with_paper_2009.csv", index=False)
    print(metrics_table.to_string(index=False))
    print(comparison_table.to_string(index=False))

    best_model_name = metrics_table.iloc[0]["model"]
    best_model = trained[best_model_name]["model"]
    print(f"  => Mo hinh tot nhat theo AUC-ROC tren test: {best_model_name}")

    print("[6/7] Thi nghiem so sanh rieng: SMOTE (khong thay the class_weight)...")
    smote_trained = run_smote_experiment(X_train_t, y_train.to_numpy(), X_val_t, y_val.to_numpy())
    resampling_comparison = build_resampling_comparison_table(
        trained, smote_trained, X_test_t, y_test.to_numpy(), X_val_t, y_val.to_numpy()
    )
    resampling_comparison.to_csv(REPORTS_DIR / "v2_smote_experiment_comparison.csv", index=False)
    print(resampling_comparison.to_string(index=False))
    print(
        "  => Luu y cot 'brier' (cang thap cang tot): neu brier cua SMOTE (chua "
        "hieu chinh) xau hon baseline, risk_score sau SMOTE da bi lech va CAN "
        "hieu chinh (calibrate) truoc khi dung cho priority_score. Pipeline "
        "chinh (buoc 7) van dung best_model tu class_weight nhu tren, khong "
        "tu dong doi model chi vi AUC/F1 cao hon o day."
    )

    print("[7/7] SHAP feature importance & bang xep hang uu tien thu hoi no...")
    try:
        from src.interpret import compute_shap_values, top_features_by_shap

        shap_values, X_sample = compute_shap_values(best_model, X_test_t, feature_names)
        top_features = top_features_by_shap(shap_values, feature_names)
        top_features.to_csv(REPORTS_DIR / "v2_shap_top_features.csv", index=False)
        print(top_features.to_string(index=False))
    except ImportError:
        print("  (bo qua SHAP: chua cai dat thu vien 'shap')")

    risk_scores = best_model.predict_proba(X_test_t)[:, 1]
    priority_ranking = build_priority_ranking(X_test, X_test.index.to_series(name="ID"), risk_scores)
    priority_ranking.to_csv(REPORTS_DIR / "v2_priority_ranking.csv", index=False)
    print(priority_ranking.head(10).to_string(index=False))

    summary = {
        "best_model": best_model_name,
        "best_model_test_auc": float(metrics_table.iloc[0]["test_auc_roc"]),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    with open(REPORTS_DIR / "v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nHoan tat (version 2). Ket qua da luu trong thu muc reports/ (tien to 'v2_').")


if __name__ == "__main__":
    main()
