const essayInput = document.getElementById("essay");
const wordCounter = document.getElementById("word-counter");
const form = document.getElementById("essay-form");
const analyzeButton = document.getElementById("analyze-button");
const buttonText = document.getElementById("button-text");


if (essayInput) {

    essayInput.addEventListener("input", function () {

        const text = this.value.trim();

        const words = text
            ? text.split(/\s+/).length
            : 0;

        wordCounter.textContent =
            `${words.toLocaleString()} words`;

    });

}


if (form) {

    form.addEventListener("submit", function () {

        analyzeButton.disabled = true;

        buttonText.textContent =
            "Analyzing essay...";

        analyzeButton.style.opacity =
            "0.75";

    });

}