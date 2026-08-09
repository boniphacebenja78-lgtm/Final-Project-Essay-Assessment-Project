import re
import pandas as pd

from services.dataset_loader import load_dataset


def clean_text(text):
    """
    Clean essay text while preserving information
    useful for essay assessment.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Replace new lines and tabs
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Keep letters, numbers and basic punctuation
    text = re.sub(
        r"[^a-z0-9.,!?;:'\"()\-\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(df):
    """
    Apply text preprocessing to the dataset.
    """

    df = df.copy()

    # Create cleaned text column
    df["clean_text"] = df["full_text"].apply(clean_text)

    # Remove empty essays
    df = df[df["clean_text"].str.len() > 0].copy()

    return df


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_dataset()

    print("\nOriginal dataset:")
    print(f"Rows: {len(df)}")

    df = preprocess_dataset(df)

    print("\nPreprocessed dataset:")
    print(f"Rows: {len(df)}")

    print("\nSample cleaned essays:")

    print(
        df[
            ["essay_id", "score", "clean_text"]
        ].head()
    )

    print("\nText preprocessing successful!")