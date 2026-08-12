"""Version 3 — tuning bằng Optuna (thay cho grid search thủ công).

File này KHÔNG sửa src/models.py, src/models_v2.py. Áp dụng Optuna cho cả 6
model (không chỉ 3 model boosting — Optuna không giới hạn loại thuật toán, xem
thảo luận trong notebook), với không gian tham số MỞ RỘNG hơn grid cố định của
v1/v2 (khoảng liên tục thay vì vài giá trị rời rạc) — đây là điểm Optuna phát
huy lợi thế thật sự so với grid search nhỏ.

Vẫn giữ nguyên: tách 1 tập validation riêng (Bước 3.3) để đánh giá mỗi trial —
mỗi trial vẫn chỉ fit 1 lần trên train, predict 1 lần trên validation (giống
v1/v2), CHỈ khác cách chọn bộ tham số kế tiếp (TPE thay vì liệt kê hết lưới).
Việc đổi sang 5-fold CV (đánh giá ổn định hơn 1 validation set) là hướng khác,
xem src/models_v4.py.

Xử lý mất cân bằng: giữ đúng logic của src/models.py / src/models_v2.py cho
từng model (class_weight, scale_pos_weight, sample_weight, oversample thủ công
cho MLP) — không đổi cách xử lý, chỉ đổi cách tìm tham số.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.models import _oversample_minority

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network", "GBM", "LightGBM"]

# So trial moi model. Grid cu (v1/v2) co 4-8 to hop; o day cho ngan sach lon
# hon vi khong gian tim kiem rong hon nhieu (them subsample, reg_alpha,
# reg_lambda, max_features...).
DEFAULT_N_TRIALS = 40


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "Logistic Regression":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "Neural Network":
        hidden = trial.suggest_categorical(
            "hidden_layer_sizes", ["32", "64", "64,32", "128,64", "128,64,32"]
        )
        return {
            "hidden_layer_sizes": tuple(int(x) for x in hidden.split(",")),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        }
    if model_name == "GBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"Unknown model: {model_name}")


def _build_estimator(model_name: str, params: dict, pos_weight: float):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
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
    if model_name == "GBM":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    model_name: str, X_train, y_train, X_val, y_val, n_trials: int = DEFAULT_N_TRIALS
):
    """Chạy 1 Optuna study cho 1 model. Trả về (best_model_đã_fit_trên_train,
    best_params, best_val_auc, trials_df) — trials_df chứa TOÀN BỘ lịch sử
    trial (không chỉ best), để kiểm tra độ ổn định / mức độ ảnh hưởng của
    từng tham số (khác với grid search thủ công ở v1/v2, vốn không lưu lại).
    """
    y_train_arr = np.asarray(y_train)
    pos_weight = (y_train_arr == 0).sum() / max((y_train_arr == 1).sum(), 1)

    fit_X, fit_y = X_train, y_train_arr
    fit_sample_weight = None
    if model_name == "Neural Network":
        fit_X, fit_y = _oversample_minority(X_train, y_train_arr, RANDOM_STATE)
    elif model_name == "GBM":
        fit_sample_weight = compute_sample_weight("balanced", y_train_arr)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        model = _build_estimator(model_name, params, pos_weight)
        if fit_sample_weight is not None:
            model.fit(fit_X, fit_y, sample_weight=fit_sample_weight)
        else:
            model.fit(fit_X, fit_y)
        val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_proba)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.insert(0, "model", model_name)

    best_params = study.best_params
    if model_name == "Neural Network":
        best_params = dict(best_params)
        best_params["hidden_layer_sizes"] = tuple(
            int(x) for x in best_params["hidden_layer_sizes"].split(",")
        )

    best_model = _build_estimator(model_name, best_params, pos_weight)
    if fit_sample_weight is not None:
        best_model.fit(fit_X, fit_y, sample_weight=fit_sample_weight)
    else:
        best_model.fit(fit_X, fit_y)

    return best_model, best_params, study.best_value, trials_df


def tune_all_models(
    X_train, y_train, X_val, y_val, n_trials: int = DEFAULT_N_TRIALS, model_names=None
) -> tuple[dict, pd.DataFrame]:
    """Tune cả 6 model bằng Optuna. Trả về (results, all_trials_df) —
    all_trials_df gộp lịch sử trial của mọi model, dùng để vẽ biểu đồ /
    kiểm tra độ ổn định (vd. optuna.visualization.plot_param_importances).
    """
    if model_names is None:
        model_names = MODEL_NAMES

    results = {}
    all_trials = []
    for model_name in model_names:
        model, params, val_auc, trials_df = tune_model(
            model_name, X_train, y_train, X_val, y_val, n_trials=n_trials
        )
        results[model_name] = {"model": model, "best_params": params, "val_auc": val_auc}
        all_trials.append(trials_df)

    return results, pd.concat(all_trials, ignore_index=True)
