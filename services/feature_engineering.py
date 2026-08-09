import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from services.dataset_loader import load_dataset
from services.text_preprocessing import preprocess_dataset


class EssayFeatureEngineer:
    """
    Creates machine-learning features from essays.

    Features:
    1. TF-IDF features
    2. Essay-level statistical features
    """

    def __init__(self, max_features=5000):
        self.max_features = max_features

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

    def create_text_features(self, texts, fit=True):
        """
        Convert essays into TF-IDF numerical features.
        """

        if fit:
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)

        return features

    @staticmethod
    def create_essay_features(df):
        """
        Create numerical features describing essay structure.
        """

        features = pd.DataFrame(index=df.index)

        text = df["clean_text"]

        # Word count
        features["word_count"] = text.apply(
            lambda x: len(x.split())
        )

        # Character count
        features["character_count"] = text.apply(
            len
        )

        # Sentence count
        features["sentence_count"] = text.apply(
            lambda x: max(
                1,
                len(
                    [
                        sentence
                        for sentence in x.split(".")
                        if sentence.strip()
                    ]
                )
            )
        )

        # Average sentence length
        features["avg_sentence_length"] = (
            features["word_count"]
            / features["sentence_count"]
        )

        # Unique word count
        features["unique_word_count"] = text.apply(
            lambda x: len(set(x.split()))
        )

        # Vocabulary richness
        features["vocabulary_richness"] = (
            features["unique_word_count"]
            / features["word_count"].replace(0, 1)
        )

        # Paragraph count
        features["paragraph_count"] = df["full_text"].apply(
            lambda x: max(
                1,
                len(
                    [
                        paragraph
                        for paragraph in str(x).split("\n")
                        if paragraph.strip()
                    ]
                )
            )
        )

        return features


def build_features(df, vectorizer=None):
    """
    Build the complete feature set.
    """

    engineer = EssayFeatureEngineer()

    if vectorizer is None:

        tfidf_features = engineer.create_text_features(
            df["clean_text"],
            fit=True
        )

    else:

        engineer.vectorizer = vectorizer

        tfidf_features = engineer.create_text_features(
            df["clean_text"],
            fit=False
        )

    essay_features = engineer.create_essay_features(df)

    return tfidf_features, essay_features, engineer


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_dataset()

    print(f"Original rows: {len(df)}")

    print("\nPreprocessing dataset...")

    df = preprocess_dataset(df)

    print(f"Preprocessed rows: {len(df)}")

    print("\nCreating features...")

    tfidf_features, essay_features, engineer = build_features(df)

    print("\nTF-IDF features:")
    print(f"Rows: {tfidf_features.shape[0]}")
    print(f"Columns: {tfidf_features.shape[1]}")

    print("\nEssay-level features:")
    print(f"Rows: {essay_features.shape[0]}")
    print(f"Columns: {essay_features.shape[1]}")

    print("\nEssay-level feature names:")

    print(
        list(essay_features.columns)
    )

    print("\nSample essay features:")

    print(
        essay_features.head()
    )

    print("\nFeature engineering successful!")