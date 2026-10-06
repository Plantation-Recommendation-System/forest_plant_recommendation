import { useState } from 'react';
import { fetchRecommendations } from './services/recommendationApi.js';
import sitePhoto from './assets/site-photo.jpg';


const HERO_IMAGES = {
  main:
    'https://images.pexels.com/photos/17023585/pexels-photo-17023585.jpeg?auto=compress&cs=tinysrgb&w=1200',

  side:
    'https://images.pexels.com/photos/1407305/pexels-photo-1407305.jpeg?auto=compress&cs=tinysrgb&w=900',

  detail:
    'https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg?auto=compress&cs=tinysrgb&w=900',
};


const WET_BEHAVIOR_OPTIONS = [
  {
    value: 'loose_gritty',
    title: 'Loose and gritty',
    description:
      'Feels sandy and does not hold together well.',
  },
  {
    value: 'weak_ball',
    title: 'Forms a soft ball',
    description:
      'Holds together lightly but breaks apart fairly easily.',
  },
  {
    value: 'sticky',
    title: 'Sticky and holds together',
    description:
      'Feels sticky and stays together when pressed between your fingers.',
  },
];


function climateLabel(value) {
  const labels = {
    A_strict_match: 'Strong climate match',
    B_near_match: 'Near climate match',
    C_partial_match: 'Partial climate match',
    D_weak_match: 'Weak climate match',
  };

  return labels[value] || value || 'Unknown';
}


function soilMatchLabel(value) {
  const labels = {
    strong_texture_match: 'Strong soil match',
    moderate_texture_match: 'Moderate soil match',
    weak_texture_match: 'Weak soil match',
    insufficient_soil_evidence: 'Limited soil evidence',
  };

  return labels[value] || value || 'Unknown';
}


function consistencyLabel(value) {
  const labels = {
    consistent: 'Consistent',
    inconsistent: 'Inconsistent',
    questionnaire_unknown: 'Not available',
  };

  return labels[value] || value || 'Not available';
}


const initialForm = {
  location: '',
  latitude: null,
  longitude: null,
  locationAccuracy: null,
  soilImage: null,
  wetBehavior: '',
};


function App() {
  const [form, setForm] = useState(initialForm);

  const [view, setView] = useState('form');

  const [recommendations, setRecommendations] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [detectingLocation, setDetectingLocation] =
    useState(false);

  const [showManualLocation, setShowManualLocation] =
    useState(false);

  const [error, setError] =
    useState('');


  function updateField(event) {
    const { name, value } = event.target;

    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  }


  function updateManualLocation(event) {
    const value = event.target.value;

    setForm((current) => ({
      ...current,

      location: value,

      latitude: null,
      longitude: null,
      locationAccuracy: null,
    }));
  }


  function updateSoilImage(event) {
    const file =
      event.target.files?.[0] || null;

    setForm((current) => ({
      ...current,
      soilImage: file,
    }));
  }


  function detectLocation() {
    setError('');

    if (!navigator.geolocation) {
      setError(
        'Location detection is not supported by this browser. Please enter the location manually.',
      );

      setShowManualLocation(true);

      return;
    }

    setDetectingLocation(true);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const {
          latitude,
          longitude,
          accuracy,
        } = position.coords;

        setForm((current) => ({
          ...current,

          location: '',

          latitude,
          longitude,

          locationAccuracy:
            accuracy,
        }));

        setShowManualLocation(false);

        setDetectingLocation(false);

        setError('');
      },

      (locationError) => {
        setDetectingLocation(false);

        setShowManualLocation(true);

        if (
          locationError.code === 1
        ) {
          setError(
            'Location permission was denied. Please allow location access or enter the location manually.',
          );

          return;
        }

        if (
          locationError.code === 2
        ) {
          setError(
            'Your location could not be detected. Please try again or enter it manually.',
          );

          return;
        }

        if (
          locationError.code === 3
        ) {
          setError(
            'Location detection timed out. Please try again or enter it manually.',
          );

          return;
        }

        setError(
          'Unable to detect your location. Please enter it manually.',
        );
      },

      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 60000,
      },
    );
  }


  async function submit(event) {
    event.preventDefault();

    const hasGps =
      form.latitude !== null &&
      form.longitude !== null;

    const hasManualLocation =
      Boolean(
        form.location.trim(),
      );

    if (
      !hasGps &&
      !hasManualLocation
    ) {
      setError(
        'Please detect your planting location or enter it manually.',
      );

      return;
    }

    if (!form.soilImage) {
      setError(
        'Please upload a soil photo.',
      );

      return;
    }

    if (!form.wetBehavior) {
      setError(
        'Please tell us how the soil feels when wet.',
      );

      return;
    }

    setLoading(true);

    setError('');

    try {
      const payload =
        await fetchRecommendations({
          location:
            form.location,

          latitude:
            form.latitude,

          longitude:
            form.longitude,

          soilImage:
            form.soilImage,

          wetBehavior:
            form.wetBehavior,
        });

      setRecommendations(
        payload,
      );

      setView(
        'results',
      );
    } catch (requestError) {
      setError(
        requestError.message,
      );
    } finally {
      setLoading(false);
    }
  }


  function resetForm() {
    setRecommendations(null);

    setView('form');

    setError('');
  }


  const hasDetectedLocation =
    form.latitude !== null &&
    form.longitude !== null;


  return (
    <div className="site-root">

      {view === 'form' ? (

        <main className="page-wrap">

          {/* =====================================
              NAVIGATION
          ====================================== */}

          <header className="topbar">

            <div className="brand-mark">

              <span>+</span>

              EcoRoot

            </div>


            <nav className="topbar-actions">

              <a
                className="topbar-chip"
                href="#site-form"
              >
                Start assessment
              </a>

            </nav>

          </header>


          {/* =====================================
              HERO
          ====================================== */}

          <section className="hero-panel">

            <div className="hero-copy">

              <p className="eyebrow">
                Environmental Plant Intelligence
              </p>


              <h1 className="hero-title">

                The right plants begin with
                the right environment.

              </h1>


              <p className="hero-subtitle">

                Climate, Soil, and Location
                Intelligence for Identifying
                Suitable Plant Species

              </p>


              <p className="hero-text">

                EcoRoot studies the planting
                location, local climate and soil
                conditions to identify plant
                species that are environmentally
                suitable for the site.

              </p>


              <div className="hero-actions">

                <a
                  className="primary-link"
                  href="#site-form"
                >
                  Begin Site Assessment
                </a>


                <div className="hero-mini-note">

                  Location
                  {' • '}
                  Soil
                  {' • '}
                  Environmental suitability

                </div>

              </div>


              <div className="hero-stats">

                <div className="stat-card">

                  <strong>
                    Climate
                  </strong>

                  <span>
                    Site-specific climate
                    suitability analysis
                  </span>

                </div>


                <div className="stat-card">

                  <strong>
                    Soil
                  </strong>

                  <span>
                    Soil image and field
                    observation analysis
                  </span>

                </div>


                <div className="stat-card">

                  <strong>
                    Suitable Plants
                  </strong>

                  <span>
                    Environmentally matched
                    plant species
                  </span>

                </div>

              </div>

            </div>


            {/* =====================================
                HERO IMAGES
            ====================================== */}

            <div className="hero-visual">

              <div
                className="
                  visual-card
                  visual-main
                "
              >

                <img
                  src={
                    HERO_IMAGES.main
                  }
                  alt="Healthy plant"
                />

              </div>


              <div className="visual-side-stack">

                <div
                  className="
                    visual-card
                    visual-small
                  "
                >

                  <img
                    src={
                      HERO_IMAGES.side
                    }
                    alt="Green leaves"
                  />

                </div>


                <div className="visual-note">

                  <span className="visual-note-label">

                    EcoRoot workflow

                  </span>


                  <strong>

                    Location → Climate →
                    Soil → Suitable Plants

                  </strong>


                  <p>

                    Environmental evidence
                    guides every visible plant
                    result.

                  </p>

                </div>

              </div>


              <div className="visual-badge">

                <img
                  src={
                    HERO_IMAGES.detail
                  }
                  alt="Plant detail"
                />


                <div>

                  <span>
                    Site-based analysis
                  </span>

                  <strong>
                    Built around real
                    environmental conditions
                  </strong>

                </div>

              </div>

            </div>

          </section>


          {/* =====================================
              ASSESSMENT
          ====================================== */}

          <section
            id="site-form"
            className="assessment-section"
          >

            <div className="section-heading">

              <p className="eyebrow">
                Site Assessment
              </p>


              <h2>

                Understand the site
                before choosing the plant.

              </h2>


              <p>

                EcoRoot only asks for information
                that directly contributes to the
                visible plant suitability
                assessment.

              </p>
                <div className="section-photo-card">
    <img
      src={sitePhoto}
      alt="Plants in a natural indoor setting"
    />
  </div>

            </div>


            <form
              className="form-card"
              onSubmit={submit}
            >

              {/* =================================
                  LOCATION
              ================================== */}

              <div className="form-block">

                <div className="form-block-header">

                  <div>

                    <p className="field-kicker">
                      Step 01
                    </p>


                    <h3>
                      Planting Location
                    </h3>

                  </div>


                  <button
                    className="ghost-button"
                    type="button"
                    onClick={
                      detectLocation
                    }
                    disabled={
                      detectingLocation
                    }
                  >

                    {
                      detectingLocation
                        ? 'Detecting...'

                        : hasDetectedLocation
                          ? 'Detect Again'

                          : 'Detect My Location'
                    }

                  </button>

                </div>


                <p className="field-help">

                  Detect the exact planting
                  location so EcoRoot can
                  automatically obtain the
                  climate conditions for
                  the site.

                </p>


                {hasDetectedLocation && (

                  <div className="info-panel">

                    <strong>
                      Location detected
                      successfully
                    </strong>


                    <span>

                      Latitude:{' '}

                      {
                        Number(
                          form.latitude,
                        ).toFixed(6)
                      }

                    </span>


                    <span>

                      Longitude:{' '}

                      {
                        Number(
                          form.longitude,
                        ).toFixed(6)
                      }

                    </span>


                    {
                      Number.isFinite(
                        Number(
                          form.locationAccuracy,
                        ),
                      ) && (

                        <span>

                          GPS accuracy:{' '}

                          {
                            Math.round(
                              Number(
                                form.locationAccuracy,
                              ),
                            )
                          }

                          {' '}meters

                        </span>

                      )
                    }

                  </div>

                )}


                {!hasDetectedLocation && (

                  <button
                    className="text-button"
                    type="button"
                    onClick={() =>
                      setShowManualLocation(
                        (current) =>
                          !current,
                      )
                    }
                  >

                    {
                      showManualLocation
                        ? 'Hide manual entry'
                        : 'Enter location manually'
                    }

                  </button>

                )}


                {showManualLocation && (

                  <label className="field">

                    <span>
                      Manual location
                    </span>


                    <input
                      name="location"
                      value={
                        form.location
                      }
                      onChange={
                        updateManualLocation
                      }
                      placeholder="Example: Pune, Maharashtra, India"
                    />


                    <small>

                      Use manual entry if GPS
                      detection is unavailable.

                    </small>

                  </label>

                )}

              </div>


              {/* =================================
                  SOIL + QUESTIONNAIRE GRID
              ================================== */}

              <div className="form-grid">

                {/* =============================
                    SOIL IMAGE
                ============================== */}

                <div className="form-block">

                  <div
                    className="
                      form-block-header
                      compact
                    "
                  >

                    <div>

                      <p className="field-kicker">
                        Step 02
                      </p>


                      <h3>
                        Soil Photo
                      </h3>

                    </div>

                  </div>


                  <p className="field-help">

                    Upload a clear close-up
                    photo of the soil from
                    the planting site.

                  </p>


                  <label className="upload-box">

                    <input
                      type="file"
                      accept="image/*"
                      onChange={
                        updateSoilImage
                      }
                    />


                    <span className="upload-title">

                      Upload Soil Image

                    </span>


                    <span className="upload-text">

                      JPG, PNG or WEBP

                    </span>

                  </label>


                  {form.soilImage && (

                    <div className="selected-file">

                      Selected file:{' '}

                      {
                        form
                          .soilImage
                          .name
                      }

                    </div>

                  )}

                </div>


                {/* =============================
                    WET SOIL
                ============================== */}

                <div className="form-block">

                  <div
                    className="
                      form-block-header
                      compact
                    "
                  >

                    <div>

                      <p className="field-kicker">
                        Step 03
                      </p>


                      <h3>
                        Wet-Soil Behavior
                      </h3>

                    </div>

                  </div>


                  <p className="field-help">

                    Wet a small amount of soil
                    and select the option that
                    describes it best.

                  </p>


                  <div className="behavior-options">

                    {
                      WET_BEHAVIOR_OPTIONS
                        .map(
                          (option) => (

                            <label
                              className={
                                `behavior-card ${
                                  form.wetBehavior ===
                                  option.value
                                    ? 'selected'
                                    : ''
                                }`
                              }
                              key={
                                option.value
                              }
                            >

                              <input
                                type="radio"
                                name="wetBehavior"
                                value={
                                  option.value
                                }
                                checked={
                                  form.wetBehavior ===
                                  option.value
                                }
                                onChange={
                                  updateField
                                }
                              />


                              <div>

                                <strong>

                                  {
                                    option.title
                                  }

                                </strong>


                                <p>

                                  {
                                    option
                                      .description
                                  }

                                </p>

                              </div>

                            </label>

                          ),
                        )
                    }

                  </div>

                </div>

              </div>


              {/* =================================
                  ERROR
              ================================== */}

              {error && (

                <p className="error-message">

                  {error}

                </p>

              )}


              {/* =================================
                  SUBMIT
              ================================== */}

              <div className="form-footer">

                <button
                  className="submit-button"
                  type="submit"
                  disabled={
                    loading ||
                    detectingLocation
                  }
                >

                  {
                    loading
                      ? 'Analyzing Site...'
                      : 'Find Suitable Plants'
                  }

                </button>


                <p className="form-note">

                  Plant suitability is determined
                  primarily from climate and soil
                  evidence for the selected site.

                </p>

              </div>

            </form>

          </section>

        </main>

      ) : (

        <Results
          payload={
            recommendations
          }
          onBack={
            resetForm
          }
        />

      )}

    </div>
  );
}


/* =========================================================
   RESULTS PAGE
========================================================= */

function Results({
  payload,
  onBack,
}) {
  const rows =
    payload?.recommendations ||
    [];

  const site =
    payload?.site ||
    {};

  const soil =
    payload?.soil_prediction ||
    {};

  const counts =
    payload?.pipeline_counts ||
    {};

  const environmentalCount =
    counts.environmental_candidates ??
    counts.environmental_count ??
    '—';

  const placeName =
    site.city ||
    site.province ||
    site.query ||
    'this planting site';

  const summaryItems = [
    {
      label: 'Soil Type',
      value:
        soil.predicted_class ||
        '—',
      icon: '🪴',
      note: 'Detected from your uploaded soil image',
    },
    {
      label: 'Soil Observation',
      value: consistencyLabel(
        soil.questionnaire_consistency,
      ),
      icon: '✅',
      note: 'Compared with your wet-soil answer',
    },
    {
      label: 'Suitable Candidates',
      value:
        environmentalCount,
      icon: '🌿',
      note: 'Plants that passed environmental screening',
    },
    {
      label: 'Plants Shown',
      value: rows.length,
      icon: '🌳',
      note: 'Final species displayed to you',
    },
  ];

  return (
    <main className="page-wrap results-page">
      <header className="topbar">
        <div className="brand-mark">
          <span>+</span>
          EcoRoot
        </div>

        <div className="topbar-actions">
          <button
            className="topbar-chip"
            type="button"
            onClick={onBack}
          >
            New Assessment
          </button>
        </div>
      </header>

      <section className="results-hero">
        <div className="results-hero-copy">
          <p className="eyebrow">
            Site Assessment Results
          </p>

          <h1 className="results-title">
            Suitable plants for {placeName}
          </h1>

          {site.display_name && (
            <p className="results-location">
              {site.display_name}
            </p>
          )}
        </div>

        <div className="results-side-note">
          <span className="results-side-kicker">
            EcoRoot Summary
          </span>

          <h3>
            Plant suggestions guided by
            climate and soil conditions
          </h3>

          <p>
            These results are based on the
            planting location, climate profile,
            uploaded soil photo, and your
            wet-soil observation. The goal is
            to show plant species that better
            fit the environmental conditions of
            the site.
          </p>
        </div>
      </section>

      <section className="summary-grid">
        {summaryItems.map((item) => (
          <article
            className="summary-card summary-card-rich"
            key={item.label}
          >
            <div className="summary-card-head">
              <div className="summary-icon">
                {item.icon}
              </div>

              <div className="summary-card-text">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            </div>

            <p className="summary-note">
              {item.note}
            </p>
          </article>
        ))}
      </section>

      <section className="result-grid">
        {rows.map((row) => {
          const soilMatch =
            Number(
              row.soil_texture_match_pct,
            );

          return (
            <article
              className="plant-card"
              key={`${row.final_recommendation_rank}-${row.project_species_resolved}`}
            >
              <div className="plant-card-top">
                <span className="rank-pill">
                  #
                  {row.final_recommendation_rank}
                </span>

                <span className="match-pill">
                  {climateLabel(
                    row.climate_tier,
                  )}
                </span>
              </div>

              <div className="plant-species-visual">
                <div className="plant-card-icon">
                  🌳
                </div>

                <div className="plant-card-icon-label">
                  Tree species
                </div>
              </div>

              <h2>
                {row.project_species_resolved}
              </h2>

              <div className="divider-line" />

              <dl className="metric-list">
                <div>
                  <dt>
                    Climate suitability
                  </dt>
                  <dd>
                    {climateLabel(
                      row.climate_tier,
                    )}
                  </dd>
                </div>

                <div>
                  <dt>
                    Soil compatibility
                  </dt>
                  <dd>
                    {Number.isFinite(
                      soilMatch,
                    )
                      ? `${soilMatch.toFixed(
                          1,
                        )}%`
                      : '—'}
                  </dd>
                </div>

                <div>
                  <dt>
                    Soil match
                  </dt>
                  <dd>
                    {soilMatchLabel(
                      row.soil_texture_match_label,
                    )}
                  </dd>
                </div>
              </dl>
            </article>
          );
        })}
      </section>

      {rows.length === 0 && (
        <p className="error-message">
          No suitable plants were returned
          for this site.
        </p>
      )}
    </main>
  );
}


export default App;