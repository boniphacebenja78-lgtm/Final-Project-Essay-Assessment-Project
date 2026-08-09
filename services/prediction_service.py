import os
import joblib

from services.text_preprocessing import clean_text


# =========================================================
# Model configuration
# =========================================================

MODEL_PATH = "models/essay_score_model.pkl"


# =========================================================
# Load model
# =========================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# =========================================================
# Predict essay score
# =========================================================

def predict_score(essay):

    if not essay or not essay.strip():

        raise ValueError(
            "Essay cannot be empty."
        )

    # Clean the essay using the same
    # preprocessing used during development
    cleaned_essay = clean_text(
        essay
    )

    # Load trained model
    model = load_model()

    # Make prediction
    prediction = model.predict(
        [cleaned_essay]
    )

    score = int(
        prediction[0]
    )

    return score


# =========================================================
# Test prediction service
# =========================================================

if __name__ == "__main__":

    sample_essay = """
    The author argues that studying Venus is important
    because it can help scientists understand the history
    of planets and the conditions that may support life.
    The evidence presented supports the main argument
    by explaining the characteristics of Venus.
    """

    print(
        "Testing prediction service..."
    )

    score = predict_score(
        sample_essay
    )

    print(
        f"Predicted score: {score}/6"
    )