# =========================================================
# REGRESSION MODEL TRAINING
# Automated Essay Scoring System
#
# Models:
# 1. Linear Regression
# 2. Random Forest Regression
# 3. Gradient Boosting Regression
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from services.dataset_loader import load_dataset
from services.text_preprocessing import preprocess_dataset
from services.feature_engineering import EssayFeatureEngineer
from services.train_test_split import split_dataset


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features():

    print("=" * 60)
    print("PREPARING DATA FOR REGRESSION MODELS")
    print("=" * 60)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = load_dataset()

    print(
        f"Dataset rows: {len(df)}"
    )

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    print("\nPreprocessing dataset...")

    df = preprocess_dataset(df)

    print(
        f"Preprocessed rows: {len(df)}"
    )

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    print("\nSplitting dataset...")

    train_df, test_df = split_dataset(df)

    print(
        f"Training essays: {len(train_df)}"
    )

    print(
        f"Testing essays: {len(test_df)}"
    )

    # -----------------------------------------------------
    # Create essay-level features
    # -----------------------------------------------------

    print("\nCreating essay-level features...")

    train_features = EssayFeatureEngineer.create_essay_features(
        train_df
    )

    test_features = EssayFeatureEngineer.create_essay_features(
        test_df
    )

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    feature_names = list(
        train_features.columns
    )

    print("\nFeatures used:")

    for feature in feature_names:
        print(
            f" - {feature}"
        )

    # -----------------------------------------------------
    # Target variable
    # -----------------------------------------------------

    y_train = train_df["score"].astype(float)
    y_test = test_df["score"].astype(float)

    # -----------------------------------------------------
    # Convert to numpy
    # -----------------------------------------------------

    X_train = train_features.values
    X_test = test_features.values

    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "Feature scaling completed."
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
        scaler
    )


# =========================================================
# TRAIN LINEAR REGRESSION
# =========================================================

def train_linear_regression(
    X_train,
    y_train
):

    print("\nTraining Linear Regression...")

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    print(
        "Linear Regression training completed."
    )

    return model


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

def train_random_forest(
    X_train,
    y_train
):

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    return model


# =========================================================
# TRAIN GRADIENT BOOSTING
# =========================================================

def train_gradient_boosting(
    X_train,
    y_train
):

    print("\nTraining Gradient Boosting...")

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Gradient Boosting training completed."
    )

    return model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(
        X_test
    )

    # Keep predictions inside the
    # valid ASAP score range.
    predictions = np.clip(
        predictions,
        1,
        6
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n")
    print("=" * 60)
    print(
        f"{model_name.upper()} RESULTS"
    )
    print("=" * 60)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model,
    filename
):

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        model,
        path
    )

    print(
        f"Saved: {path}"
    )

    return path


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train_all_models():

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        scaler
    ) = prepare_features()

    # -----------------------------------------------------
    # Train models
    # -----------------------------------------------------

    linear_model = train_linear_regression(
        X_train,
        y_train
    )

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    gradient_boosting_model = train_gradient_boosting(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    linear_results = evaluate_model(
        linear_model,
        X_test,
        y_test,
        "Linear Regression"
    )

    random_forest_results = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    gradient_results = evaluate_model(
        gradient_boosting_model,
        X_test,
        y_test,
        "Gradient Boosting"
    )

    # -----------------------------------------------------
    # Create comparison table
    # -----------------------------------------------------

    results = pd.DataFrame([
        linear_results,
        random_forest_results,
        gradient_results
    ])

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save models
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    save_model(
        linear_model,
        "linear_regression_model.pkl"
    )

    save_model(
        random_forest_model,
        "random_forest_model.pkl"
    )

    save_model(
        gradient_boosting_model,
        "gradient_boosting_model.pkl"
    )

    # -----------------------------------------------------
    # Save scaler
    # -----------------------------------------------------

    scaler_path = os.path.join(
        MODEL_DIR,
        "regression_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"Saved: {scaler_path}"
    )

    # -----------------------------------------------------
    # Save feature names
    # -----------------------------------------------------

    feature_path = os.path.join(
        MODEL_DIR,
        "regression_feature_names.pkl"
    )

    joblib.dump(
        feature_names,
        feature_path
    )

    print(
        f"Saved: {feature_path}"
    )

    # -----------------------------------------------------
    # Save evaluation results
    # -----------------------------------------------------

    results_path = os.path.join(
        MODEL_DIR,
        "regression_model_results.pkl"
    )

    joblib.dump(
        results.to_dict(
            orient="records"
        ),
        results_path
    )

    print(
        f"Saved: {results_path}"
    )

    # -----------------------------------------------------
    # Display best model
    # -----------------------------------------------------

    best_model = results.sort_values(
        "rmse"
    ).iloc[0]

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"RMSE: {best_model['rmse']:.4f}"
    )

    print(
        f"MAE: {best_model['mae']:.4f}"
    )

    print(
        f"R²: {best_model['r2']:.4f}"
    )

    print("\nRegression training completed successfully!")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    train_all_models()