import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = "used_car_prices_new_250.csv"
MODEL_FILE = "used_car_price_model.pkl"
METRICS_FILE = "metrics.json"

TARGET = "Selling_Price"

CATEGORICAL_COLUMNS = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
]

NUMERIC_COLUMNS = [
    "Year",
    "Kilometers_Driven",
    "Engine_CC",
    "Seats",
]


def train_model():
    print("Loading used-car dataset...")

    df = pd.read_csv(DATA_FILE)

    required_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET]

    if not set(required_columns).issubset(df.columns):
        raise ValueError("Dataset is missing required columns.")

    df = df[required_columns].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df = df.dropna(subset=CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

    if len(df) < 10:
        raise ValueError("Not enough valid records to train the model.")

    X = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            ),
        ]
    )

    print("Training Random Forest regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "dataset_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
    }

    joblib.dump(model, MODEL_FILE)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel evaluation")
    print("----------------")
    print(f"Records used: {len(df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")


if __name__ == "__main__":
    train_model()
