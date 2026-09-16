/* =========================================================
   PLANT SURVIVAL AI
   Recommendation Results
   ========================================================= */


/* =========================================================
   ELEMENTS
   ========================================================= */

const recommendationGrid =
    document.getElementById(
        "recommendation-grid"
    );

const emptyResults =
    document.getElementById(
        "empty-results"
    );

const resultLocation =
    document.getElementById(
        "result-location"
    );


/* =========================================================
   LOAD SPECIES INFORMATION
   ========================================================= */

async function loadSpeciesData() {

    const response =
        await fetch("species.json");

    if (!response.ok) {

        throw new Error(
            "Unable to load species information."
        );
    }

    return await response.json();
}


/* =========================================================
   GOOGLE IMAGE URL
   ========================================================= */

function getGoogleImagesUrl(scientificName) {

    const query =
        encodeURIComponent(
            scientificName + " tree"
        );

    return (
        "https://www.google.com/search" +
        "?tbm=isch&q=" +
        query
    );
}


/* =========================================================
   DISPLAY SCORE
   ========================================================= */

function getDisplayScore(value) {

    const number =
        parseFloat(value);

    if (!Number.isFinite(number)) {
        return null;
    }


    /*
     * V7 is frozen and currently stores the
     * final score at approximately 100x the
     * human-facing 0-100 scale.
     *
     * We only convert it for display.
     */

    return number / 100;
}


/* =========================================================
   EVIDENCE LABEL
   ========================================================= */

function getEvidenceLabel(score) {

    const value =
        parseFloat(score);


    if (!Number.isFinite(value)) {
        return "Unavailable";
    }


    if (value >= 75) {
        return "Strong";
    }


    if (value >= 50) {
        return "Moderate";
    }


    return "Limited";
}


/* =========================================================
   CREATE RECOMMENDATION CARD
   ========================================================= */

function createRecommendationCard(
    recommendation,
    speciesData,
    index
) {

    const normalizedName =
        (
            recommendation.species_normalized ||
            ""
        )
        .trim()
        .toLowerCase();


    const species =
        speciesData[normalizedName] || {};


    const scientificName =
        species.scientific_name ||
        recommendation.species_normalized ||
        "Unknown species";


    const localName =
        species.local_name ||
        "Local name unavailable";


    const commonName =
        species.common_name ||
        "";


    const finalScore =
        getDisplayScore(
            recommendation.final_score
        );


    const v5Score =
        parseFloat(
            recommendation.v5_survival_score
        );


    const regionalScore =
        parseFloat(
            recommendation.regional_evidence_score
        );


    const googleImagesUrl =
        getGoogleImagesUrl(
            scientificName
        );


    const card =
        document.createElement("article");

    card.className =
        "recommendation-card";


    /* -----------------------------------------------------
       Image area
       ----------------------------------------------------- */

    const imageArea =
        document.createElement("div");

    imageArea.className =
        "tree-image";


    /*
     * Image placeholder for now.
     *
     * Later we will connect Wikimedia / GBIF
     * images here.
     */

    imageArea.innerHTML = `

        <div class="image-placeholder">

            <div class="placeholder-tree">
                🌳
            </div>

            <span>
                Tree image
            </span>

        </div>

    `;


    /* -----------------------------------------------------
       Card content
       ----------------------------------------------------- */

    const content =
        document.createElement("div");

    content.className =
        "tree-card-content";


    const rank =
        document.createElement("div");

    rank.className =
        "tree-rank";

    rank.textContent =
        "#" + (
            index + 1
        );


    const local =
        document.createElement("h2");

    local.className =
        "tree-local-name";

    local.textContent =
        localName;


    const scientific =
        document.createElement("div");

    scientific.className =
        "tree-scientific";

    scientific.textContent =
        scientificName;


    const common =
        document.createElement("div");

    common.className =
        "tree-common";

    common.textContent =
        commonName;


    /* -----------------------------------------------------
       Score
       ----------------------------------------------------- */

    const scoreBox =
        document.createElement("div");

    scoreBox.className =
        "score-box";


    scoreBox.innerHTML = `

        <div class="score-label">
            Suitability
        </div>

        <div class="score-value">
            ${
                finalScore !== null
                    ? finalScore.toFixed(1)
                    : "—"
            }

            <span>/ 100</span>
        </div>

    `;


    /* -----------------------------------------------------
       Score details
       ----------------------------------------------------- */

    const details =
        document.createElement("div");

    details.className =
        "tree-details";


    details.innerHTML = `

        <div class="detail-row">

            <span>
                Survival suitability
            </span>

            <strong>
                ${
                    Number.isFinite(v5Score)
                        ? v5Score.toFixed(1)
                        : "—"
                }
            </strong>

        </div>


        <div class="detail-row">

            <span>
                Regional evidence
            </span>

            <strong>
                ${getEvidenceLabel(regionalScore)}
            </strong>

        </div>

    `;


    /* -----------------------------------------------------
       Recommendation reasons
       ----------------------------------------------------- */

    const reasons =
        document.createElement("div");

    reasons.className =
        "tree-reasons";


    reasons.innerHTML = `

        <div class="reason-title">
            Why it is recommended
        </div>

        <div class="reason">
            <span>✓</span>
            Supported by the survival model
        </div>

        <div class="reason">
            <span>✓</span>
            Regional occurrence evidence
        </div>

    `;


    /* -----------------------------------------------------
       Actions
       ----------------------------------------------------- */

    const actions =
        document.createElement("div");

    actions.className =
        "tree-actions";


    const imagesLink =
        document.createElement("a");

    imagesLink.className =
        "images-link";

    imagesLink.href =
        googleImagesUrl;

    imagesLink.target =
        "_blank";

    imagesLink.rel =
        "noopener noreferrer";

    imagesLink.textContent =
        "See more images ↗";


    actions.appendChild(
        imagesLink
    );


    /* -----------------------------------------------------
       Assemble
       ----------------------------------------------------- */

    content.appendChild(rank);

    content.appendChild(local);

    content.appendChild(scientific);

    if (commonName) {
        content.appendChild(common);
    }

    content.appendChild(scoreBox);

    content.appendChild(details);

    content.appendChild(reasons);

    content.appendChild(actions);


    card.appendChild(imageArea);

    card.appendChild(content);


    return card;
}


/* =========================================================
   MAIN
   ========================================================= */

async function loadRecommendations() {

    try {

        const stored =
            sessionStorage.getItem(
                "plantSurvivalRecommendations"
            );


        const location =
            sessionStorage.getItem(
                "plantSurvivalLocation"
            );


        if (location) {

            resultLocation.textContent =
                "Recommendations for " +
                location;
        }


        if (!stored) {

            throw new Error(
                "No recommendation data found."
            );
        }


        const data =
            JSON.parse(stored);


        if (
            !data.success ||
            !Array.isArray(
                data.recommendations
            ) ||
            data.recommendations.length === 0
        ) {

            throw new Error(
                "No recommendations available."
            );
        }


        const speciesData =
            await loadSpeciesData();


        recommendationGrid.innerHTML =
            "";


        data.recommendations.forEach(
            function (
                recommendation,
                index
            ) {

                const card =
                    createRecommendationCard(
                        recommendation,
                        speciesData,
                        index
                    );


                recommendationGrid.appendChild(
                    card
                );

            }
        );


    } catch (error) {

        console.error(
            "Results error:",
            error
        );


        recommendationGrid.style.display =
            "none";

        emptyResults.style.display =
            "block";

    }

}


/* =========================================================
   START
   ========================================================= */

loadRecommendations();