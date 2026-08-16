import os
import joblib
import pandas as pd

from services.essay_analyzer import analyze_essay


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_DIR = "models"

GRADIENT_BOOSTING_MODEL = os.path.join(
    MODEL_DIR,
    "gradient_boosting_model.pkl"
)

RANDOM_FOREST_MODEL = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

LINEAR_REGRESSION_MODEL = os.path.join(
    MODEL_DIR,
    "linear_regression_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "regression_scaler.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "regression_feature_names.pkl"
)


# =========================================================
# LOAD MODELS
# =========================================================

def load_regression_models():

    if not os.path.exists(GRADIENT_BOOSTING_MODEL):

        raise FileNotFoundError(
            "Gradient Boosting model not found at: "
            f"{GRADIENT_BOOSTING_MODEL}"
        )

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            "Regression scaler not found at: "
            f"{SCALER_PATH}"
        )

    if not os.path.exists(FEATURE_NAMES_PATH):

        raise FileNotFoundError(
            "Regression feature names not found at: "
            f"{FEATURE_NAMES_PATH}"
        )

    gradient_boosting = joblib.load(
        GRADIENT_BOOSTING_MODEL
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    return (
        gradient_boosting,
        scaler,
        feature_names
    )


# =========================================================
# CREATE FEATURES FROM ESSAY
# =========================================================

def create_prediction_features(
    essay,
    feature_names
):

    analysis = analyze_essay(
        essay
    )

    # -----------------------------------------------------
    # Convert essay analysis into DataFrame
    # -----------------------------------------------------

    features = {

        "word_count":
            analysis["word_count"],

        "character_count":
            analysis["character_count"],

        "sentence_count":
            analysis["sentence_count"],

        "avg_sentence_length":
            analysis["average_sentence_length"],

        "unique_word_count":
            analysis["unique_word_count"],

        "vocabulary_richness":
            analysis["vocabulary_richness"],

        "paragraph_count":
            analysis["paragraph_count"]
    }

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    feature_df = pd.DataFrame(
        [features]
    )

    # -----------------------------------------------------
    # Make sure the feature order is exactly
    # the same as during model training
    # -----------------------------------------------------

    feature_df = feature_df[
        feature_names
    ]

    return feature_df


# =========================================================
# PREDICT SCORE
# =========================================================

def predict_score(essay):

    if not essay or not essay.strip():

        raise ValueError(
            "Essay cannot be empty."
        )

    # -----------------------------------------------------
    # Load trained regression model
    # -----------------------------------------------------

    (
        gradient_boosting,
        scaler,
        feature_names
    ) = load_regression_models()

    # -----------------------------------------------------
    # Create numerical essay features
    # -----------------------------------------------------

    feature_df = create_prediction_features(
        essay,
        feature_names
    )

    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    scaled_features = scaler.transform(
        feature_df.values
    )

    # -----------------------------------------------------
    # Predict using Gradient Boosting
    #
    # Gradient Boosting had the best performance:
    #
    # MAE  = 0.5006
    # RMSE = 0.6577
    # R²   = 0.5968
    # -----------------------------------------------------

    prediction = gradient_boosting.predict(
        scaled_features
    )

    # -----------------------------------------------------
    # Convert prediction to valid 1–6 range
    # -----------------------------------------------------

    score = float(
        prediction[0]
    )

    score = max(
        1.0,
        min(
            6.0,
            score
        )
    )

    # -----------------------------------------------------
    # Round to 2 decimal places
    # -----------------------------------------------------

    score = round(
        score,
        2
    )

    return score


# =========================================================
# GET ESSAY FEATURES
# =========================================================

def get_essay_features(essay):

    if not essay or not essay.strip():

        raise ValueError(
            "Essay cannot be empty."
        )

    (
        _,
        _,
        feature_names
    ) = load_regression_models()

    features = create_prediction_features(
        essay,
        feature_names
    )

    return features.iloc[0].to_dict()


# =========================================================
# TEST PREDICTION SERVICE
# =========================================================

if __name__ == "__main__":

    sample_essay = """
    Technology has changed the way students learn.
    Students can now access information quickly through
    online resources and educational applications.

    However, technology can also create distractions.
    Students may spend more time on social media instead
    of concentrating on their academic work.

    In conclusion, technology can be very useful for
    education when students use it responsibly.
    """

    print("=" * 60)
    print("TESTING GRADIENT BOOSTING PREDICTION SERVICE")
    print("=" * 60)

    score = predict_score(
        sample_essay
    )

    print()
    print("Prediction successful!")
    print()
    print(
        f"Predicted Score: {score}/6"
    )

    print()
    print("Essay Features:")

    features = get_essay_features(
        sample_essay
    )

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )