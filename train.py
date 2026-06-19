"""
train.py  —  Indian House Price Prediction: Model Training
Trains Linear Regression, Random Forest, Gradient Boosting.
Saves best model + all pipelines to models/model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load data ───────────────────────────────────────────────────────────────
DATA_PATH = "data/house_data.csv"
if not os.path.exists(DATA_PATH):
    print("[ERROR] data/house_data.csv not found. Run generate_data.py first!")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")

FEATURES = ["Area", "Bedrooms", "Bathrooms", "Parking", "Age",
            "Furnishing", "Floor", "City"]
TARGET   = "Price"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
numeric_features      = ["Area", "Bedrooms", "Bathrooms", "Parking", "Age"]
categorical_features  = ["Furnishing", "Floor", "City"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                       numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ── 3. Models ──────────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results, trained_pipelines = {}, {}

print("\n[INFO] Training & Evaluating Models ...\n" + "-" * 60)
for name, regressor in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv}
    trained_pipelines[name] = pipe

    print(f"  {name}")
    print(f"    MAE   = Rs. {mae:.2f} Lakh")
    print(f"    RMSE  = Rs. {rmse:.2f} Lakh")
    print(f"    R2    = {r2:.4f}")
    print(f"    CV R2 = {cv:.4f}")
    print()

# ── 4. Best model ──────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["CV_R2"])
best_pipe = trained_pipelines[best_name]
print(f"[BEST] {best_name}  (CV R2 = {results[best_name]['CV_R2']:.4f})")

# ── 5. Feature importances ─────────────────────────────────────────────────────
try:
    fi  = best_pipe.named_steps["regressor"].feature_importances_
    enc = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    enc_names = enc.get_feature_names_out(categorical_features)
    feat_names = numeric_features + list(enc_names)
    fi_df = (pd.DataFrame({"Feature": feat_names, "Importance": fi})
               .sort_values("Importance", ascending=False))
    print("\n[INFO] Top 12 Feature Importances:")
    print(fi_df.head(12).to_string(index=False))
except AttributeError:
    pass

# ── 6. Save ────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
payload = {
    "model":          best_pipe,
    "model_name":     best_name,
    "all_results":    results,
    "features":       FEATURES,
    "all_pipelines":  trained_pipelines,
}
joblib.dump(payload, "models/model.pkl")
print("\n[OK] Saved -> models/model.pkl")
