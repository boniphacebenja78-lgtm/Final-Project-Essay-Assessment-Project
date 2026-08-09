import pandas as pd

from services.dataset_loader import load_dataset


def prepare_dataset():

    df = load_dataset()

    print("\nOriginal dataset:")
    print("Rows:", len(df))

    # Keep only the columns required for our assessment system
    df = df[
        [
            "essay_id",
            "score",
            "full_text",
            "assignment",
            "prompt_name"
        ]
    ].copy()

    # Remove missing essays
    df = df.dropna(
        subset=["full_text", "score"]
    )

    # Convert essay text to string
    df["full_text"] = df["full_text"].astype(str)

    # Remove empty essays
    df = df[
        df["full_text"].str.strip().str.len() > 0
    ]

    # Convert score to numeric
    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce"
    )

    # Remove rows with invalid scores
    df = df.dropna(
        subset=["score"]
    )

    # Remove duplicate essay IDs
    df = df.drop_duplicates(
        subset=["essay_id"]
    )

    # Reset dataframe index
    df = df.reset_index(drop=True)

    print("\nPrepared dataset:")
    print("Rows:", len(df))

    print("\nScore distribution:")
    print(
        df["score"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":

    df = prepare_dataset()

    print("\nDataset preparation successful!")

    print("\nSample essays:")
    print(
        df[
            [
                "essay_id",
                "score",
                "full_text"
            ]
        ].head()
    )