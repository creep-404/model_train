"""
Car Price Prediction Model
===========================
Trains a regression model to predict New Price (EGP) of cars listed on
an Egyptian car marketplace, using engineered features from the listing text.
"""

import re
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
df = pd.read_csv("car_price.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

print(f"Raw rows: {len(df)}")

# ----------------------------------------------------------------------
# 2. CLEAN NUMERIC / DATE COLUMNS
# ----------------------------------------------------------------------
def money_to_num(s):
    if pd.isna(s):
        return np.nan
    return float(re.sub(r"[^\d.]", "", str(s)) or np.nan)

df["old_price"] = df["Old Price"].apply(money_to_num)
df["new_price"] = df["New Price"].apply(money_to_num)
df["date_range"] = pd.to_datetime(df["date_range"], errors="coerce")

# price_change: sign (up/down) + magnitude
df["change_direction"] = df["Price Change"].apply(
    lambda s: "up" if "trending_up" in str(s) else ("down" if "trending_down" in str(s) else "flat")
)
df["change_amount"] = df["Price Change"].apply(
    lambda s: float(re.sub(r"[^\d.]", "", str(s)) or 0) * (-1 if "trending_down" in str(s) else 1)
)

# ----------------------------------------------------------------------
# 3. PARSE THE "Classes" LISTING STRING
#    e.g. "Toyota Corolla A/T / High Line / Hybrid 2023"
# ----------------------------------------------------------------------
TRANS_PATTERN = r"(A/T|Automtic|Automatic|M/T|Manual|manual)"

def parse_classes(text):
    text = str(text)
    year_match = re.search(r"(19|20)\d{2}\s*$", text)
    year = int(year_match.group()) if year_match else np.nan

    trans_match = re.search(TRANS_PATTERN, text, flags=re.IGNORECASE)
    transmission_raw = trans_match.group() if trans_match else None
    if transmission_raw:
        transmission = "Automatic" if transmission_raw.lower() in ("a/t", "automtic", "automatic") else "Manual"
    else:
        transmission = "Unknown"

    brand = text.split()[0] if text.split() else None

    # everything before the transmission token = "Brand Model"
    if trans_match:
        pre = text[:trans_match.start()].strip()
    else:
        pre = text[:year_match.start()].strip() if year_match else text.strip()
    model = " ".join(pre.split()[1:]) if len(pre.split()) > 1 else pre

    return pd.Series([brand, model, transmission, year])

df[["brand", "model", "transmission", "year"]] = df["Classes"].apply(parse_classes)

# Trim/version = leftover words after transmission token, before year (rough signal of trim level)
def parse_trim(text):
    text = str(text)
    trans_match = re.search(TRANS_PATTERN, text, flags=re.IGNORECASE)
    year_match = re.search(r"(19|20)\d{2}\s*$", text)
    if trans_match and year_match:
        trim = text[trans_match.end():year_match.start()]
    elif year_match:
        trim = text[:year_match.start()]
    else:
        trim = ""
    trim = trim.replace("/", " ").strip()
    return trim if trim else "Base"

df["trim"] = df["Classes"].apply(parse_trim)

# date-derived features
df["listing_year"] = df["date_range"].dt.year
df["listing_month"] = df["date_range"].dt.month
df["car_age_at_listing"] = df["listing_year"] - df["year"]

# ----------------------------------------------------------------------
# 4. DROP UNUSABLE ROWS
# ----------------------------------------------------------------------
before = len(df)
df = df.dropna(subset=["new_price", "old_price", "year", "brand"])
df = df[df["new_price"] > 0]
print(f"Dropped {before - len(df)} rows with missing/invalid critical fields -> {len(df)} rows remain")

# Cap extremely rare brands into "Other" to avoid one-hot blowup / overfitting
brand_counts = df["brand"].value_counts()
rare_brands = brand_counts[brand_counts < 5].index
df["brand"] = df["brand"].where(~df["brand"].isin(rare_brands), "Other")

# Same for trim (very high cardinality, free text)
trim_counts = df["trim"].value_counts()
rare_trims = trim_counts[trim_counts < 5].index
df["trim_grouped"] = df["trim"].where(~df["trim"].isin(rare_trims), "Other")

print(f"Unique brands: {df['brand'].nunique()}, unique trims (grouped): {df['trim_grouped'].nunique()}")

# ----------------------------------------------------------------------
# 5. FEATURES / TARGET
# ----------------------------------------------------------------------
# We predict New Price from car identity (brand, model line, trim, year,
# transmission) plus listing timing and the prior (old) price -- i.e. "what
# will the next listed price be given this car and its last known price".
feature_cols_numeric = ["year", "car_age_at_listing", "old_price", "listing_month"]
feature_cols_categorical = ["brand", "transmission", "trim_grouped", "change_direction"]

X = df[feature_cols_numeric + feature_cols_categorical].copy()
y = df["new_price"].copy()

# impute remaining stray NaNs (e.g. car_age_at_listing) with median
X["car_age_at_listing"] = X["car_age_at_listing"].fillna(X["car_age_at_listing"].median())
X["listing_month"] = X["listing_month"].fillna(X["listing_month"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ----------------------------------------------------------------------
# 6. PREPROCESS + MODEL PIPELINE
# ----------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
    ]
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

    results[name] = {"MAE": mae, "MAPE": mape, "R2": r2, "CV_R2_mean": cv_r2}
    fitted_pipelines[name] = pipe
    print(f"\n{name}")
    print(f"  MAE  : {mae:,.0f} EGP")
    print(f"  MAPE : {mape:.2%}")
    print(f"  R2   : {r2:.4f}")
    print(f"  CV R2 (5-fold, train): {cv_r2:.4f}")

# ----------------------------------------------------------------------
# 7. PICK BEST MODEL
# ----------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best_pipe = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

joblib.dump(best_pipe, "car_price_model.joblib")

with open("model_results.json", "w") as f:
     plt.savefig("model_evaluation.png", dpi=140)

# ----------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (for tree models) + PRED VS ACTUAL PLOT
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pred vs actual
preds_best = best_pipe.predict(X_test)
axes[0].scatter(y_test, preds_best, alpha=0.3, s=10)
lims = [0, max(y_test.max(), preds_best.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1)
axes[0].set_xlabel("Actual New Price (EGP)")
axes[0].set_ylabel("Predicted New Price (EGP)")
axes[0].set_title(f"{best_name}: Predicted vs Actual")

# Feature importance if available
if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
    ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(feature_cols_categorical)
    all_names = feature_cols_numeric + list(cat_names)
    importances = best_pipe.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=all_names).sort_values(ascending=False).head(15)
    axes[1].barh(imp_series.index[::-1], imp_series.values[::-1])
    axes[1].set_title("Top 15 Feature Importances")
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "N/A for this model", ha="center")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=140)
print("\nSaved: car_price_model.joblib, model_results.json, model_evaluation.png")
