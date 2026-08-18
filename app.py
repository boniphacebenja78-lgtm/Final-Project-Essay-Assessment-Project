from flask import Flask, render_template, request
import os

from pypdf import PdfReader
from docx import Document

from services.prediction_service import predict_score
from services.feedback_service import generate_feedback


# =========================================================
# Flask application
# =========================================================

app = Flask(__name__)


# =========================================================
# Configuration
# =========================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# =========================================================
# Home page
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# Extract text from TXT
# =========================================================

def extract_txt(file):

    content = file.read()

    try:

        return content.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        return content.decode(
            "latin-1"
        )


# =========================================================
# Extract text from PDF
# =========================================================

def extract_pdf(file):

    text = []

    reader = PdfReader(
        file
    )

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text.append(
                page_text
            )

    return "\n\n".join(
        text
    )


# =========================================================
# Extract text from DOCX
# =========================================================

def extract_docx(file):

    document = Document(
        file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                text
            )

    return "\n\n".join(
        paragraphs
    )


# =========================================================
# Extract essay text from uploaded file
# Supports:
# TXT
# PDF
# DOCX
# =========================================================

def extract_essay_from_file(file):

    if not file:

        return ""


    filename = (
        file.filename or ""
    ).lower()


    if not filename:

        return ""


    extension = os.path.splitext(
        filename
    )[1]


    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    if extension == ".txt":

        return extract_txt(
            file
        )


    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if extension == ".pdf":

        return extract_pdf(
            file
        )


    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    if extension == ".docx":

        return extract_docx(
            file
        )


    raise ValueError(
        "Unsupported file type. "
        "Please upload a PDF, DOCX, or TXT file."
    )


# =========================================================
# Assess essay
# =========================================================

@app.route(
    "/assess",
    methods=["POST"]
)
def assess():

    # -----------------------------------------------------
    # Get pasted essay
    # -----------------------------------------------------

    essay = request.form.get(
        "essay",
        ""
    ).strip()


    # -----------------------------------------------------
    # Get uploaded file
    # -----------------------------------------------------

    uploaded_file = request.files.get(
        "essay_file"
    )


    try:

        # =================================================
        # HANDLE COMPUTER UPLOAD
        # =================================================

        if uploaded_file and uploaded_file.filename:

            uploaded_text = (
                extract_essay_from_file(
                    uploaded_file
                )
            )

            if uploaded_text:

                essay = uploaded_text.strip()


        # =================================================
        # VALIDATE ESSAY
        # =================================================

        if not essay:

            return render_template(
                "index.html",

                error=(
                    "Please paste an essay "
                    "or upload a PDF, DOCX, or TXT file."
                )
            )


        # =================================================
        # PREDICT SCORE
        # =================================================

        score = predict_score(
            essay
        )


        # =================================================
        # GENERATE FEEDBACK
        # =================================================

        feedback = generate_feedback(
            essay,
            score
        )


        # =================================================
        # DISPLAY RESULT
        # =================================================

        return render_template(
            "result.html",

            score=score,

            essay=essay,

            feedback=feedback
        )


    except Exception as error:

        return render_template(
            "index.html",

            error=f"Error assessing essay: {error}"
        )


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False
    )