
import re
import math
from collections import Counter


# =========================================================
# BASIC TEXT UTILITIES
# =========================================================

def split_sentences(text):
    """
    Split an essay into sentences.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_words(text):
    """
    Extract words from the essay.
    """

    return re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text.lower()
    )


def count_syllables(word):
    """
    Simple syllable estimation.
    This is not a perfect linguistic parser,
    but works well for readability estimation.
    """

    word = word.lower()

    vowels = "aeiouy"

    count = 0
    previous_was_vowel = False

    for character in word:

        is_vowel = character in vowels

        if is_vowel and not previous_was_vowel:
            count += 1

        previous_was_vowel = is_vowel

    # Remove silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def calculate_readability(
    words,
    sentences
):
    """
    Calculate an approximate Flesch Reading Ease score.
    """

    if not words or not sentences:
        return 0

    total_syllables = sum(
        count_syllables(word)
        for word in words
    )

    average_words_per_sentence = (
        len(words) / len(sentences)
    )

    average_syllables_per_word = (
        total_syllables / len(words)
    )

    score = (
        206.835
        - (1.015 * average_words_per_sentence)
        - (84.6 * average_syllables_per_word)
    )

    return round(
        max(0, min(100, score)),
        1
    )


# =========================================================
# VOCABULARY ANALYSIS
# =========================================================

def analyze_vocabulary(words):
    """
    Analyze vocabulary diversity.
    """

    if not words:
        return {
            "unique_words": 0,
            "vocabulary_diversity": 0,
            "repeated_words": []
        }

    word_counts = Counter(words)

    unique_words = len(
        set(words)
    )

    vocabulary_diversity = (
        unique_words / len(words)
    ) * 100

    # Ignore very common short words
    ignored_words = {
        "the",
        "and",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "that",
        "this",
        "it",
        "as",
        "by",
        "be",
        "was",
        "were",
        "or",
        "at",
        "from",
        "but",
        "they",
        "their",
        "there",
        "which",
        "have",
        "has",
        "had"
    }

    repeated_words = [
        {
            "word": word,
            "count": count
        }

        for word, count in word_counts.most_common()

        if count >= 4
        and word not in ignored_words
        and len(word) > 3
    ]

    return {
        "unique_words": unique_words,

        "vocabulary_diversity": round(
            vocabulary_diversity,
            1
        ),

        "repeated_words":
            repeated_words[:10]
    }


# =========================================================
# PARAGRAPH ANALYSIS
# =========================================================

def analyze_paragraphs(text):

    paragraphs = [
        paragraph.strip()

        for paragraph in re.split(
            r"\n\s*\n",
            text
        )

        if paragraph.strip()
    ]

    return paragraphs


# =========================================================
# BASIC STRUCTURE ANALYSIS
# =========================================================

def analyze_structure(
    text,
    paragraphs,
    sentences
):

    strengths = []

    improvements = []

    paragraph_count = len(
        paragraphs
    )

    sentence_count = len(
        sentences
    )


    # -----------------------------------------------------
    # Paragraph structure
    # -----------------------------------------------------

    if paragraph_count >= 3:

        strengths.append(
            "The essay is divided into multiple paragraphs, which helps organize the ideas."
        )

    elif paragraph_count == 2:

        improvements.append(
            "Consider separating the essay into clearer introduction, body, and conclusion paragraphs."
        )

    else:

        improvements.append(
            "The essay would benefit from clearer paragraph organization."
        )


    # -----------------------------------------------------
    # Sentence structure
    # -----------------------------------------------------

    if sentence_count >= 5:

        strengths.append(
            "The essay develops its ideas across several sentences rather than relying on very short responses."
        )

    else:

        improvements.append(
            "Develop the ideas further by providing more complete explanations and supporting details."
        )


    # -----------------------------------------------------
    # Introduction
    # -----------------------------------------------------

    first_words = text.lower()[:300]

    introduction_indicators = [
        "this essay",
        "the purpose",
        "this paper",
        "in this essay",
        "the author",
        "this article",
        "the topic"
    ]

    has_introduction_indicator = any(
        indicator in first_words

        for indicator in introduction_indicators
    )


    if has_introduction_indicator:

        strengths.append(
            "The introduction appears to establish the topic and direction of the essay."
        )

    else:

        improvements.append(
            "Make the introduction clearer by establishing the topic and main argument early."
        )


    # -----------------------------------------------------
    # Conclusion
    # -----------------------------------------------------

    last_part = text.lower()[-400:]

    conclusion_indicators = [
        "in conclusion",
        "to conclude",
        "overall",
        "therefore",
        "in summary",
        "finally"
    ]

    has_conclusion_indicator = any(
        indicator in last_part

        for indicator in conclusion_indicators
    )


    if has_conclusion_indicator:

        strengths.append(
            "The essay appears to include a concluding section that reinforces the discussion."
        )

    else:

        improvements.append(
            "Consider adding a stronger conclusion that summarizes the main argument."
        )


    return strengths, improvements


# =========================================================
# WRITING QUALITY ANALYSIS
# =========================================================

def analyze_writing_quality(
    words,
    sentences
):

    strengths = []

    improvements = []


    if not words:

        return strengths, improvements


    average_sentence_length = (
        len(words) / max(
            len(sentences),
            1
        )
    )


    # -----------------------------------------------------
    # Sentence length
    # -----------------------------------------------------

    if 10 <= average_sentence_length <= 25:

        strengths.append(
            "The average sentence length is generally suitable for academic writing."
        )

    elif average_sentence_length > 30:

        improvements.append(
            "Some sentences may be too long. Consider breaking complex sentences into shorter, clearer statements."
        )

    else:

        improvements.append(
            "Consider combining some short sentences to develop ideas more fully."
        )


    # -----------------------------------------------------
    # Vocabulary diversity
    # -----------------------------------------------------

    unique_words = len(
        set(words)
    )

    diversity = (
        unique_words / len(words)
    )


    if diversity >= 0.55:

        strengths.append(
            "The essay demonstrates a reasonably varied vocabulary."
        )

    elif diversity >= 0.40:

        improvements.append(
            "Try using a wider range of vocabulary to avoid repeating the same words."
        )

    else:

        improvements.append(
            "The vocabulary is quite repetitive. Use more precise and varied academic language."
        )


    return strengths, improvements


# =========================================================
# OVERALL ASSESSMENT
# =========================================================

def generate_overall_assessment(
    score,
    word_count,
    readability
):

    if score >= 6:

        assessment = (
            "The essay demonstrates a very strong overall performance. "
            "It appears to communicate its ideas effectively and provides "
            "a solid basis for a high assessment score."
        )

    elif score >= 5:

        assessment = (
            "The essay demonstrates strong overall performance. "
            "The main ideas are reasonably developed, although there "
            "may still be areas where clarity, evidence, or organization "
            "could be improved."
        )

    elif score >= 4:

        assessment = (
            "The essay demonstrates a satisfactory level of performance. "
            "The main ideas are present, but further development, clearer "
            "organization, and stronger supporting evidence could improve "
            "the overall quality."
        )

    elif score >= 3:

        assessment = (
            "The essay demonstrates some understanding of the topic, "
            "but several areas require improvement. The response would "
            "benefit from clearer development of ideas, stronger organization, "
            "and more precise writing."
        )

    elif score >= 2:

        assessment = (
            "The essay shows limited development of the topic. "
            "More detailed explanation, supporting evidence, and clearer "
            "organization are needed."
        )

    else:

        assessment = (
            "The essay requires significant improvement. "
            "The response should focus on developing the main argument, "
            "providing relevant supporting details, and improving structure."
        )


    # Add length observation

    if word_count < 100:

        assessment += (
            " The essay is relatively short, so additional development "
            "of ideas may be necessary."
        )

    elif word_count >= 300:

        assessment += (
            " The essay contains enough text for a more detailed development "
            "of the topic."
        )


    # Add readability observation

    if readability >= 70:

        assessment += (
            " The writing is generally easy to read."
        )

    elif readability < 40:

        assessment += (
            " Some sentences may be difficult to follow and could be simplified."
        )


    return assessment


# =========================================================
# MAIN FEEDBACK FUNCTION
# =========================================================

def generate_feedback(
    essay,
    score
):

    if not essay or not essay.strip():

        raise ValueError(
            "Essay cannot be empty."
        )


    essay = essay.strip()


    # -----------------------------------------------------
    # Basic measurements
    # -----------------------------------------------------

    words = split_words(
        essay
    )

    sentences = split_sentences(
        essay
    )

    paragraphs = analyze_paragraphs(
        essay
    )


    word_count = len(
        words
    )

    sentence_count = len(
        sentences
    )

    paragraph_count = len(
        paragraphs
    )


    # -----------------------------------------------------
    # Readability
    # -----------------------------------------------------

    readability = calculate_readability(
        words,
        sentences
    )


    # -----------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------

    vocabulary = analyze_vocabulary(
        words
    )


    # -----------------------------------------------------
    # Structure
    # -----------------------------------------------------

    structure_strengths, structure_improvements = (
        analyze_structure(
            essay,
            paragraphs,
            sentences
        )
    )


    # -----------------------------------------------------
    # Writing quality
    # -----------------------------------------------------

    writing_strengths, writing_improvements = (
        analyze_writing_quality(
            words,
            sentences
        )
    )


    # -----------------------------------------------------
    # Combine strengths
    # -----------------------------------------------------

    strengths = (
        structure_strengths
        + writing_strengths
    )


    # -----------------------------------------------------
    # Combine improvements
    # -----------------------------------------------------

    improvements = (
        structure_improvements
        + writing_improvements
    )


    # -----------------------------------------------------
    # Vocabulary feedback
    # -----------------------------------------------------

    diversity = vocabulary[
        "vocabulary_diversity"
    ]


    if diversity >= 55:

        strengths.append(
            "The essay uses a reasonably diverse range of words."
        )

    elif diversity < 40:

        improvements.append(
            "Increase vocabulary variety and avoid repeating the same expressions."
        )


    # -----------------------------------------------------
    # Repeated words
    # -----------------------------------------------------

    repeated_words = vocabulary[
        "repeated_words"
    ]


    if repeated_words:

        repeated_names = ", ".join(
            item["word"]
            for item in repeated_words[:5]
        )

        improvements.append(
            f"Some words appear frequently, including: {repeated_names}. "
            "Consider using more varied wording where appropriate."
        )


    # -----------------------------------------------------
    # Make sure we have useful feedback
    # -----------------------------------------------------

    if not strengths:

        strengths.append(
            "The essay contains a recognizable attempt to develop the topic."
        )


    if not improvements:

        improvements.append(
            "Continue refining clarity, supporting evidence, and academic expression."
        )


    # -----------------------------------------------------
    # Overall assessment
    # -----------------------------------------------------

    overall = generate_overall_assessment(
        score,
        word_count,
        readability
    )


    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = []


    for improvement in improvements[:4]:

        recommendations.append(
            improvement
        )


    recommendations.append(
        "Proofread the final essay carefully before submission."
    )


    # -----------------------------------------------------
    # Return complete feedback
    # -----------------------------------------------------

    return {

        "overall": overall,

        "strengths": strengths[:5],

        "improvements": improvements[:6],

        "recommendations":
            recommendations[:5],

        "statistics": {

            "word_count":
                word_count,

            "sentence_count":
                sentence_count,

            "paragraph_count":
                paragraph_count,

            "unique_words":
                vocabulary["unique_words"],

            "vocabulary_diversity":
                vocabulary["vocabulary_diversity"],

            "readability":
                readability,

            "average_sentence_length":
                round(
                    word_count /
                    max(
                        sentence_count,
                        1
                    ),
                    1
                )

        },

        "repeated_words":
            repeated_words[:5]

    }


# =========================================================
# TEST THE FEEDBACK SERVICE
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


    sample_score = 4


    print("\n")
    print("=" * 60)
    print("ESSAY FEEDBACK SERVICE TEST")
    print("=" * 60)


    feedback = generate_feedback(
        sample_essay,
        sample_score
    )


    print("\nOVERALL:")
    print(feedback["overall"])


    print("\nSTRENGTHS:")

    for item in feedback["strengths"]:

        print(
            f"• {item}"
        )


    print("\nAREAS FOR IMPROVEMENT:")

    for item in feedback["improvements"]:

        print(
            f"• {item}"
        )


    print("\nRECOMMENDATIONS:")

    for item in feedback["recommendations"]:

        print(
            f"• {item}"
        )


    print("\nSTATISTICS:")

    for key, value in feedback[
        "statistics"
    ].items():

        print(
            f"{key}: {value}"
        )


    print("\n")

