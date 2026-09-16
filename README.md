# Plant Survival AI

A clean rebuild centered on one source dataset: `data/survival_ml_final.csv`.

## Current Architecture

```text
React + Vite frontend
        |
        | Axios / REST
        v
FastAPI backend
        |
        v
Main survival dataset
        |
        v
Historical survival summary
```

## Project Structure

```text
AI-Based_Plantation_Recommendation_System/
├── frontend/                 React/Vite application
├── backend/                  FastAPI REST API
├── data/
│   └── survival_ml_final.csv
└── ml_pipeline/              Reserved for the rebuilt ML pipeline
```

## Main Dataset

`data/survival_ml_final.csv` is the only retained dataset. It contains historical survival observations and includes the target column `survival_per`.

Important current factors include:

- country and province
- disturbance and forest condition
- competition removal
- shading, soil preparation, water regulation, fertilisation, and protection
- planting density and planting species count
- initial age and height
- species, genus, family
- latitude and longitude
- monitoring duration
- wood-density features
- treatment
- `survival_per`, the observed survival target

## Removed From This Rebuild

The following branches were intentionally removed:

- Pune candidate pool and Pune-specific matching
- train/validation/test split files
- pollution capability data
- soil images and soil model files
- GBIF regional discovery outputs
- V6/V7 generated ranking outputs
- old ranking/evidence/soil/pollution/Pune Python scripts

## Future Data Additions

The rebuild can later add new columns or related tables for:

- climate
- rainfall and precipitation
- temperature extremes
- humidity
- evapotranspiration
- longitude/latitude-derived geographic features
- soil chemistry
- site management conditions

These should be added deliberately to the main modeling contract instead of joining unrelated local datasets.

## Run

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env -Force
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Frontend, in a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The current FastAPI endpoint reads the main CSV and returns a historical-survival summary. It is not yet a live climate-aware ML inference endpoint.
