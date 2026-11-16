
document.addEventListener("DOMContentLoaded", function () {

    const searchInput = document.getElementById("symptomSearch");
    const symptomOptions = document.querySelectorAll(".symptom-option");
    const selectedContainer = document.getElementById("selectedSymptoms");
    const selectedCount = document.getElementById("selectedCount");
    const clearButton = document.getElementById("clearButton");
    const noResults = document.getElementById("noResults");

    function updateSelectedSymptoms() {

        selectedContainer.innerHTML = "";

        const checkedSymptoms = document.querySelectorAll(
            'input[name="symptoms"]:checked'
        );

        selectedCount.textContent = checkedSymptoms.length;

        if (checkedSymptoms.length === 0) {
            selectedContainer.innerHTML =
                '<span class="empty-selection">No symptoms selected yet.</span>';
            return;
        }

        checkedSymptoms.forEach(function (checkbox) {

            const tag = document.createElement("span");
            tag.className = "symptom-tag";

            const displayName = checkbox.value
                .replaceAll("_", " ")
                .replace(/\b\w/g, letter => letter.toUpperCase());

            const name = document.createElement("span");
            name.textContent = displayName;

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.textContent = "×";
            removeButton.setAttribute("aria-label", "Remove " + displayName);

            removeButton.addEventListener("click", function () {
                checkbox.checked = false;
                updateSelectedSymptoms();
            });

            tag.appendChild(name);
            tag.appendChild(removeButton);

            selectedContainer.appendChild(tag);
        });
    }


    // Update selected symptoms whenever a checkbox changes
    symptomOptions.forEach(function (option) {

        const checkbox = option.querySelector("input");

        checkbox.addEventListener("change", updateSelectedSymptoms);

    });


    // Search and filter symptoms
    searchInput.addEventListener("input", function () {

        const searchTerm = searchInput.value.toLowerCase().trim();

        let visibleCount = 0;

        symptomOptions.forEach(function (option) {

            const symptomName = option.dataset.symptom;

            const matches = symptomName.includes(searchTerm);

            option.style.display = matches ? "flex" : "none";

            if (matches) {
                visibleCount++;
            }

        });

        noResults.hidden = visibleCount !== 0;

    });


    // Clear all selected symptoms
    clearButton.addEventListener("click", function () {

        document.querySelectorAll(
            'input[name="symptoms"]'
        ).forEach(function (checkbox) {
            checkbox.checked = false;
        });

        updateSelectedSymptoms();

    });


    // Initialize selected symptoms on page load
    updateSelectedSymptoms();


    // Scroll to result after a successful prediction
    const result = document.getElementById("predictionResult");

    if (result) {
        result.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

});
