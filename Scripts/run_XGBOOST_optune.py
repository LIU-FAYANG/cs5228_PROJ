import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("../Utils")
from ML_training_utils_tools import evaluate_model

datapath = "../Dataset/"
df_train = pd.read_csv(datapath + "train_data_for_modeling(no_standardization).csv")
print(f"Successfully loaded train data, the shape is {df_train.shape}")

with open(datapath + "all_final_features.json", "r") as f:
    all_final_features = json.load(f)

feature_cols = [
    col for col in all_final_features if col not in ["RESALE_PRICE", "LOG_RESALE_PRICE"]
]

X = df_train[feature_cols]
y = df_train["RESALE_PRICE"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Successfully split the data")
print(
    f"The shape of train data is {X_train.shape},the shape of valid data is {X_valid.shape}"
)

import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# 2) Optuna objective (5-fold CV with early stopping on GPU)
def objective(trial):
    params = {
        "n_estimators": 10_000,  # rely on early stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 100.0, log=True
        ),
        "early_stopping_rounds": 200,
        "eval_metric": "rmse",
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 100.0, log=True),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        # GPU settings
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        "n_jobs": 1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmses = []

    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        booster = model.get_booster()
        dval = DMatrix(X_va)
        preds = booster.predict(dval, iteration_range=(0, model.best_iteration + 1))

        fold_rmses.append(mean_squared_error(y_va, preds))

    return float(np.mean(fold_rmses))


import time

# 2a. Using Optuna to search for best hyperparameters

N_TRIALS = 200  # adjust up if you have more GPU time
BASELINE_RMSE = baseline_rmse

print("Optuna GPU Search -- Smart exploration")
print("Start the Optuna Search...")
start_time = time.time()

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

elapsed_time = time.time() - start_time
print(f"\n Optuna Search completed! Time taken: {elapsed_time/60:.1f} minutes")

best_params_optuna = study.best_params.copy()
print(f"\nBest parameters (Optuna):")
for k in sorted(best_params_optuna.keys()):
    print(f"  {k}: {best_params_optuna[k]}")

# ---- Final refit on CPU (save memory), still using early stopping ----
final_params_cpu = best_params_optuna.copy()
final_params_cpu.update(
    {
        "n_estimators": 10_000,
        "tree_method": "hist",  # CPU
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "early_stopping_rounds": 200,
    }
)

xgb_final = XGBRegressor(**final_params_cpu)
xgb_final.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

y_pred_val_optuna = xgb_final.predict(
    X_valid, iteration_range=(0, xgb_final.best_iteration + 1)
)

# If you have your evaluate_model helper defined, use it for identical printout:
optuna_rmse, optuna_mae, optuna_r2 = evaluate_model(
    y_valid, y_pred_val_optuna, "Optuna Search - Validation"
)

print(f"\nImprovement: ${BASELINE_RMSE - optuna_rmse:,.2f} (vs Baseline)")


import pandas as pd, json, pathlib as pl

out = pl.Path("../Models/XGBoost_notebook")
out.mkdir(parents=True, exist_ok=True)

cv_xgb = pd.DataFrame(study.trials_dataframe())  # 或 rs_xgb.cv_results_
cv_xgb.to_csv(out / "xgb_cv_results.csv", index=False)
(out / "xgb_best_params.json").write_text(json.dumps(study.best_params, indent=2))
