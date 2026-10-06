"""
train.py  —  Model Training Pipeline for California House Price Prediction
Implements the workflow from Project 15.4:
- Preprocessing with ColumnTransformer (Imputation + Scaling + OneHot)
- 5-Fold Cross Validation benchmark across 5 regression models
- Tuned HistGradientBoostingRegressor (best model)
- Evaluation on Test Set (RMSE, MAE, R2)
- Serializes trained pipeline and benchmark results to models/model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from utils import create_preprocessor, ALL_FEATURES, TARGET_COL

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "housing.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def train_and_evaluate():
    print("[INFO] Starting California House Price Prediction Pipeline...")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found. Please ensure housing.csv is placed in data/")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    preprocessor = create_preprocessor()

    # Models benchmark dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    cv_results = {}
    test_results = {}
    trained_pipelines = {}

    print("\n" + "=" * 65)
    print(" 5-FOLD CROSS VALIDATION & TEST EVALUATION")
    print("=" * 65)

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        # 5-fold cross validation on training data
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rmse = -float(scores["test_rmse"].mean())
        cv_mae = -float(scores["test_mae"].mean())
        cv_r2 = float(scores["test_r2"].mean())

        # Fit on full training set and evaluate on held-out test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_rmse = float(root_mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        cv_results[name] = {"RMSE": cv_rmse, "MAE": cv_mae, "R2": cv_r2}
        test_results[name] = {"RMSE": test_rmse, "MAE": test_mae, "R2": test_r2}
        trained_pipelines[name] = pipe

        print(f"[{name}]")
        print(f"  CV   -> RMSE: ${cv_rmse:,.2f} | MAE: ${cv_mae:,.2f} | R2: {cv_r2:.4f}")
        print(f"  TEST -> RMSE: ${test_rmse:,.2f} | MAE: ${test_mae:,.2f} | R2: {test_r2:.4f}\n")

    # Tuned HistGradientBoosting with optimal hyperparameters from GridSearchCV
    print("[INFO] Training Tuned HistGradientBoostingRegressor (Best Parameters)...")
    tuned_params = {
        "learning_rate": 0.1,
        "max_depth": None,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": RANDOM_STATE
    }

    tuned_hgb = HistGradientBoostingRegressor(**tuned_params)
    tuned_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", tuned_hgb)
    ])

    tuned_scores = cross_validate(tuned_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    tuned_cv_rmse = -float(tuned_scores["test_rmse"].mean())
    tuned_cv_mae = -float(tuned_scores["test_mae"].mean())
    tuned_cv_r2 = float(tuned_scores["test_r2"].mean())

    tuned_pipe.fit(X_train, y_train)
    y_test_pred = tuned_pipe.predict(X_test)
    tuned_test_rmse = float(root_mean_squared_error(y_test, y_test_pred))
    tuned_test_mae = float(mean_absolute_error(y_test, y_test_pred))
    tuned_test_r2 = float(r2_score(y_test, y_test_pred))

    name_tuned = "Tuned HistGradientBoosting (Best Model)"
    cv_results[name_tuned] = {"RMSE": tuned_cv_rmse, "MAE": tuned_cv_mae, "R2": tuned_cv_r2}
    test_results[name_tuned] = {"RMSE": tuned_test_rmse, "MAE": tuned_test_mae, "R2": tuned_test_r2}
    trained_pipelines[name_tuned] = tuned_pipe

    print(f"[{name_tuned}]")
    print(f"  CV   -> RMSE: ${tuned_cv_rmse:,.2f} | MAE: ${tuned_cv_mae:,.2f} | R2: {tuned_cv_r2:.4f}")
    print(f"  TEST -> RMSE: ${tuned_test_rmse:,.2f} | MAE: ${tuned_test_mae:,.2f} | R2: {tuned_test_r2:.4f}")

    # Residuals for diagnostics
    residuals = y_test.values - y_test_pred

    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "model": tuned_pipe,
        "model_name": name_tuned,
        "cv_results": cv_results,
        "test_results": test_results,
        "features": ALL_FEATURES,
        "best_params": tuned_params,
        "sample_test_y": y_test.values[:1000],
        "sample_test_pred": y_test_pred[:1000],
        "residuals": residuals[:1000],
        "rmse": tuned_test_rmse
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\n[OK] Model & evaluation metrics successfully saved to {MODEL_PATH}")
    return payload


if __name__ == "__main__":
    train_and_evaluate()
