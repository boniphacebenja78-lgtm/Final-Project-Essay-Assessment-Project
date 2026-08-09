import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from services.dataset_loader import load_dataset
from services.text_preprocessing import preprocess_dataset
from services.train_test_split import split_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "essay_score_model.pkl"
)


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

def prepare_data():

    print("Loading dataset...")

    df = load_dataset()

    print(f"Original dataset: {len(df)} rows")

    print("\nPreprocessing dataset...")

    df = preprocess_dataset(df)

    print(f"Preprocessed dataset: {len(df)} rows")

    print("\nSplitting dataset...")

    train_df, test_df = split_dataset(df)

    print(f"Training essays: {len(train_df)}")
    print(f"Testing essays: {len(test_df)}")

    return train_df, test_df


# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------

def create_model():

    model = Pipeline([
        
        (
            "tfidf",

            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return model


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model():

    train_df, test_df = prepare_data()

    X_train = train_df["clean_text"]

    y_train = train_df["score"]

    print("\nCreating model...")

    model = create_model()

    print("Model created successfully.")

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("\nModel training completed!")

    return model, train_df, test_df


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

def save_model(model):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully to:"
    )

    print(MODEL_PATH)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    model, train_df, test_df = train_model()

    save_model(model)

    print("\n===================================")
    print("MODEL TRAINING SUCCESSFUL!")
    print("===================================")

    print(
        f"\nTraining samples: {len(train_df)}"
    )

    print(
        f"Testing samples: {len(test_df)}"
    )