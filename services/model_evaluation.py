import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from services.dataset_loader import load_dataset
from services.text_preprocessing import preprocess_dataset
from services.train_test_split import split_dataset


MODEL_PATH = "models/essay_score_model.pkl"


def evaluate_model():

    print("Loading trained model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first."
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    # -----------------------------------------------------
    # Load and prepare dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = load_dataset()

    print(f"Dataset rows: {len(df)}")

    print("\nPreprocessing dataset...")

    df = preprocess_dataset(df)

    # -----------------------------------------------------
    # Recreate the same train/test split
    # -----------------------------------------------------

    print("\nCreating test set...")

    train_df, test_df = split_dataset(df)

    X_test = test_df["clean_text"]
    y_test = test_df["score"]

    print(f"Test samples: {len(test_df)}")

    # -----------------------------------------------------
    # Make predictions
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully!")

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print(matrix)

    print("\n")
    print("Model evaluation completed successfully!")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix
    }


if __name__ == "__main__":

    evaluate_model()