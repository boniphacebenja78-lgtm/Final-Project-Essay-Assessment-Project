import pandas as pd
import os


DATASET_PATH = os.path.join(
    "data",
    "ASAP2_train_sourcetexts.csv"
)


def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("\nDataset loaded successfully!")
    print("--------------------------------")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())