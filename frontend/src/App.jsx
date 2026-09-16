import { useState } from 'react';
import { fetchRecommendations } from './services/recommendationApi.js';

const initialForm = {
  location: '',
  water: '',
  sunlight: '',
  maintenance: '',
  siteType: '',
  purpose: '',
};

const options = {
  water: ['low', 'medium', 'high'],
  sunlight: ['low', 'medium', 'high'],
  maintenance: ['low', 'medium', 'high'],
  siteType: ['roadside', 'residential', 'park', 'industrial', 'school', 'agricultural'],
  purpose: ['shade', 'pollution', 'biodiversity', 'fruit', 'beautification', 'timber', 'restoration'],
};

const labels = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  roadside: 'Roadside',
  residential: 'Residential area',
  park: 'Park / Garden',
  industrial: 'Industrial area',
  school: 'School / Campus',
  agricultural: 'Agricultural / Open land',
  shade: 'Shade',
  pollution: 'Pollution reduction',
  biodiversity: 'Biodiversity',
  fruit: 'Fruit / Food',
  beautification: 'Beautification',
  timber: 'Timber',
  restoration: 'Ecological restoration',
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [view, setView] = useState('form');
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  function updateField(event) {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  }

  function detectLocation() {
    if (!navigator.geolocation) {
      setError('Location detection is not supported by this browser.');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      ({ coords }) => setForm((current) => ({
        ...current,
        location: `${coords.latitude.toFixed(5)}, ${coords.longitude.toFixed(5)}`,
      })),
      () => setError('Unable to detect your location. Enter it manually.'),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 },
    );
  }

  async function submit(event) {
    event.preventDefault();
    if (!form.location.trim()) {
      setError('Please enter your location.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const payload = await fetchRecommendations({
        ...form,
        site_type: form.siteType,
      });
      sessionStorage.setItem('plantSurvivalUserInputs', JSON.stringify(form));
      sessionStorage.setItem('plantSurvivalRecommendations', JSON.stringify(payload));
      setRecommendations(payload);
      setView('results');
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><span>+</span> Plant Survival AI</div>
        <div className="topbar-note">Evidence-led planting decisions</div>
      </header>

      {view === 'form' ? (
        <main className="page-grid">
          <section className="intro-panel">
            <p className="eyebrow">Plantation intelligence / 01</p>
            <h1>Find a tree that belongs in its environment.</h1>
            <p className="intro-copy">A survival-data interface for exploring historical species performance and planting conditions.</p>
            <div className="signal-list">
              <span>Historical survival</span>
              <span>Species factors</span>
              <span>Geographic context</span>
            </div>
          </section>

          <form className="form-panel" onSubmit={submit}>
            <div className="form-heading">
              <p className="eyebrow">Site profile</p>
              <h2>Describe where the tree will grow.</h2>
            </div>

            <label className="field field-wide">
              <span>Location</span>
              <div className="location-row">
                <input name="location" value={form.location} onChange={updateField} placeholder="City or latitude, longitude" />
                <button className="secondary-button" type="button" onClick={detectLocation}>Detect</button>
              </div>
            </label>

            <div className="field-grid">
              {Object.entries(options).map(([name, values]) => (
                <label className="field" key={name}>
                  <span>{name === 'siteType' ? 'Site type' : name[0].toUpperCase() + name.slice(1)}</span>
                  <select name={name} value={form[name]} onChange={updateField}>
                    <option value="">Choose {name === 'siteType' ? 'site type' : name}</option>
                    {values.map((value) => <option value={value} key={value}>{labels[value]}</option>)}
                  </select>
                </label>
              ))}
            </div>

            {error && <p className="error-message">{error}</p>}

            <button className="primary-button" disabled={loading} type="submit">
              {loading ? 'Loading recommendations...' : 'Find best trees'} <span>↗</span>
            </button>
            <p className="form-note">The current API summarizes the final historical survival dataset.</p>
          </form>
        </main>
      ) : (
        <Results recommendations={recommendations} location={form.location} onBack={() => setView('form')} />
      )}
    </div>
  );
}

function Results({ recommendations, location, onBack }) {
  const rows = recommendations?.recommendations || [];
  return (
    <main className="results-page">
      <div className="results-heading">
        <div><p className="eyebrow">Recommendation set</p><h1>Supported species for {location || 'this site'}.</h1></div>
        <button className="secondary-button" type="button" onClick={onBack}>New profile</button>
      </div>
      <div className="results-grid">
        {rows.map((row, index) => {
          const score = Number(row.final_score);
          const displayScore = Number.isFinite(score) ? score / 100 : null;
          return (
            <article className="result-card" key={`${row.species_normalized}-${index}`}>
              <div className="card-topline"><span>0{index + 1}</span><span>Historical data</span></div>
              <div className="tree-symbol">♧</div>
              <h2>{row.scientific_name || row.species_normalized || 'Unknown species'}</h2>
              <p className="score">{displayScore === null ? '—' : displayScore.toFixed(1)} <small>/ 100</small></p>
              <dl>
                <div><dt>Observed survival mean</dt><dd>{Number.isFinite(Number(row.observed_survival_mean)) ? Number(row.observed_survival_mean).toFixed(1) : '—'}</dd></div>
                <div><dt>Historical observations</dt><dd>{row.survival_observations || '—'}</dd></div>
              </dl>
            </article>
          );
        })}
      </div>
      {rows.length === 0 && <p className="error-message">No supported recommendations were returned.</p>}
    </main>
  );
}

export default App;
