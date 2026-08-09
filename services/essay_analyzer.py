import re
import statistics


def analyze_essay(text):
    """
    Analyze basic linguistic and structural
    characteristics of an essay.
    """

    if not text or not text.strip():
        raise ValueError("Essay cannot be empty.")

    # --------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------

    text = text.strip()

    # Words
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    # Sentences
    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Paragraphs
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    word_count = len(words)

    character_count = len(text)

    sentence_count = max(
        1,
        len(sentences)
    )

    paragraph_count = max(
        1,
        len(paragraphs)
    )

    unique_words = set(
        word.lower()
        for word in words
    )

    unique_word_count = len(
        unique_words
    )

    vocabulary_richness = (
        unique_word_count / word_count
        if word_count > 0
        else 0
    )

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )

    # --------------------------------------------------
    # Sentence length analysis
    # --------------------------------------------------

    sentence_lengths = []

    for sentence in sentences:

        sentence_words = re.findall(
            r"\b[a-zA-Z]+\b",
            sentence
        )

        if sentence_words:
            sentence_lengths.append(
                len(sentence_words)
            )

    if sentence_lengths:

        longest_sentence = max(
            sentence_lengths
        )

        shortest_sentence = min(
            sentence_lengths
        )

    else:

        longest_sentence = 0
        shortest_sentence = 0

    # --------------------------------------------------
    # Return analysis
    # --------------------------------------------------

    return {

        "word_count": word_count,

        "character_count": character_count,

        "sentence_count": sentence_count,

        "paragraph_count": paragraph_count,

        "unique_word_count": unique_word_count,

        "vocabulary_richness": round(
            vocabulary_richness,
            3
        ),

        "average_sentence_length": round(
            average_sentence_length,
            2
        ),

        "longest_sentence": longest_sentence,

        "shortest_sentence": shortest_sentence
    }


if __name__ == "__main__":

    sample = """
    The author argues that studying Venus is important.
    Scientists can learn more about planetary environments.
    """

    result = analyze_essay(sample)

    print("\nEssay Analysis")
    print("=" * 40)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )