/* =========================================================
   PLANT SURVIVAL AI
   Main Frontend JavaScript
   ========================================================= */


/* =========================================================
   ELEMENT REFERENCES
   ========================================================= */

const soilInput =
    document.getElementById("soil-images");

const imagePreview =
    document.getElementById("image-preview");

const locationButton =
    document.getElementById("detect-location");

const locationInput =
    document.getElementById("location");

const locationStatus =
    document.getElementById("location-status");

const recommendButton =
    document.getElementById("recommend-button");


/* =========================================================
   SOIL IMAGE PREVIEW
   ========================================================= */

if (soilInput && imagePreview) {

    soilInput.addEventListener(
        "change",
        function () {

            imagePreview.innerHTML = "";

            const selectedFiles =
                Array.from(this.files);

            if (selectedFiles.length > 3) {

                alert(
                    "Please select a maximum of 3 soil images."
                );

                this.value = "";

                return;
            }


            selectedFiles.forEach(
                function (file) {

                    /*
                     * Only allow image files.
                     */

                    if (
                        !file.type.startsWith(
                            "image/"
                        )
                    ) {
                        return;
                    }


                    const reader =
                        new FileReader();


                    reader.onload =
                        function (event) {

                            const image =
                                document.createElement(
                                    "img"
                                );


                            image.src =
                                event.target.result;


                            image.className =
                                "preview-image";


                            image.alt =
                                "Selected soil image";


                            imagePreview.appendChild(
                                image
                            );

                        };


                    reader.readAsDataURL(file);

                }
            );

        }
    );

}


/* =========================================================
   LOCATION DETECTION
   ========================================================= */

if (
    locationButton &&
    locationInput &&
    locationStatus
) {

    locationButton.addEventListener(
        "click",
        function () {

            /*
             * Check browser support.
             */

            if (
                !navigator.geolocation
            ) {

                locationStatus.textContent =
                    "Location detection is not supported by your browser.";

                return;
            }


            /*
             * Show loading message.
             */

            locationStatus.textContent =
                "Detecting your location...";


            locationButton.disabled =
                true;


            /*
             * Request browser location.
             */

            navigator.geolocation.getCurrentPosition(

                function (position) {

                    const latitude =
                        position.coords.latitude;

                    const longitude =
                        position.coords.longitude;


                    /*
                     * Store coordinates in
                     * the location input.
                     */

                    locationInput.value =
                        `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`;


                    locationStatus.textContent =
                        "Location detected successfully.";


                    locationButton.disabled =
                        false;

                },


                function (error) {

                    console.error(
                        "Location error:",
                        error
                    );


                    locationStatus.textContent =
                        "Unable to detect location. Please enter it manually.";


                    locationButton.disabled =
                        false;

                },


                {
                    enableHighAccuracy: true,

                    timeout: 10000,

                    maximumAge: 300000
                }

            );

        }
    );

}


/* =========================================================
   GET FORM VALUES
   ========================================================= */

function getFormData() {

    return {

        location:
            locationInput
                ? locationInput.value.trim()
                : "",

        water:
            document.getElementById(
                "water"
            )?.value || "",

        sunlight:
            document.getElementById(
                "sunlight"
            )?.value || "",

        maintenance:
            document.getElementById(
                "maintenance"
            )?.value || "",

        siteType:
            document.getElementById(
                "site-type"
            )?.value || "",

        purpose:
            document.getElementById(
                "purpose"
            )?.value || ""

    };

}


/* =========================================================
   SAVE USER INPUTS
   ========================================================= */

function saveUserInputs(formData) {

    sessionStorage.setItem(
        "plantSurvivalUserInputs",
        JSON.stringify(formData)
    );

}


/* =========================================================
   FIND BEST TREES
   ========================================================= */

if (recommendButton) {

    recommendButton.addEventListener(
        "click",
        async function () {

            /*
             * Collect form information.
             */

            const formData =
                getFormData();


            /*
             * Basic validation.
             */

            if (!formData.location) {

                alert(
                    "Please enter your location."
                );

                if (locationInput) {
                    locationInput.focus();
                }

                return;
            }


            /*
             * Disable button while loading.
             */

            recommendButton.disabled =
                true;


            const buttonText =
                recommendButton.querySelector(
                    "span"
                );


            if (buttonText) {

                buttonText.textContent =
                    "Finding suitable trees...";
            }


            try {

                /*
                 * Call PHP API.
                 *
                 * The PHP API currently reads
                 * the existing V7 recommendation file.
                 */

                const response =
                    await fetch(
                        "api.php",
                        {
                            method: "GET",

                            headers: {
                                "Accept":
                                    "application/json"
                            },

                            cache: "no-store"
                        }
                    );


                /*
                 * Check HTTP status.
                 */

                if (!response.ok) {

                    throw new Error(
                        `Server returned HTTP ${response.status}`
                    );

                }


                /*
                 * Convert response to JSON.
                 */

                const data =
                    await response.json();


                console.log(
                    "V7 API response:",
                    data
                );


                /*
                 * Check API success.
                 */

                if (
                    !data ||
                    data.success !== true
                ) {

                    throw new Error(
                        data?.error ||
                        "Unable to generate recommendations."
                    );

                }


                /*
                 * Check that recommendations
                 * were returned.
                 */

                if (
                    !Array.isArray(
                        data.recommendations
                    ) ||
                    data.recommendations.length === 0
                ) {

                    throw new Error(
                        "No recommendations were returned."
                    );

                }


                /*
                 * Save V7 recommendation data.
                 */

                sessionStorage.setItem(
                    "plantSurvivalRecommendations",
                    JSON.stringify(data)
                );


                /*
                 * Save location.
                 */

                sessionStorage.setItem(
                    "plantSurvivalLocation",
                    formData.location
                );


                /*
                 * Save all user inputs.
                 */

                saveUserInputs(
                    formData
                );


                /*
                 * Save timestamp.
                 */

                sessionStorage.setItem(
                    "plantSurvivalRecommendationTime",
                    new Date().toISOString()
                );


                /*
                 * Go to results page.
                 */

                window.location.href =
                    "results.html";

            }


            catch (error) {

                console.error(
                    "Recommendation error:",
                    error
                );


                alert(
                    "Unable to load recommendations.\n\n" +
                    error.message
                );

            }


            finally {

                /*
                 * Restore button.
                 *
                 * This only matters if the user
                 * remains on the current page.
                 */

                recommendButton.disabled =
                    false;


                if (buttonText) {

                    buttonText.textContent =
                        "Find Best Trees";
                }

            }

        }
    );

}


/* =========================================================
   PAGE LOAD
   ========================================================= */

console.log(
    "Plant Survival AI frontend loaded successfully."
);