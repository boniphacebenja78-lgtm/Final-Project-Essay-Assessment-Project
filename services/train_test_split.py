import pandas as pd

from sklearn.model_selection import train_test_split

from services.dataset_loader import load_dataset
from services.text_preprocessing import preprocess_dataset


def split_dataset(df, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets.

    Stratification ensures that the score distribution
    remains approximately the same in both sets.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["score"]
    )

    return train_df, test_df


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_dataset()

    print(f"Original dataset: {len(df)} rows")

    print("\nPreprocessing dataset...")

    df = preprocess_dataset(df)

    print(f"Preprocessed dataset: {len(df)} rows")

    print("\nSplitting dataset...")

    train_df, test_df = split_dataset(df)

    print("\nDataset split successful!")

    print(f"Training essays: {len(train_df)}")
    print(f"Testing essays: {len(test_df)}")

    print("\nTraining score distribution:")
    print(train_df["score"].value_counts().sort_index())

    print("\nTesting score distribution:")
    print(test_df["score"].value_counts().sort_index())

    print("\nTraining percentage:")
    print(
        f"{len(train_df) / len(df) * 100:.2f}%"
    )

    print("\nTesting percentage:")
    print(
        f"{len(test_df) / len(df) * 100:.2f}%"
    )

    print("\nTrain/test split successful!")