# FastAPI Backend

The backend currently reads only `data/survival_ml_cleaned.csv`.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env -Force
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Endpoints:

- `GET /api/v1/health`
- `POST /api/v1/recommendations`
- `GET /docs`

The current recommendation endpoint aggregates historical `survival_per` by normalized species and returns the highest observed survival summaries. It is intentionally not presented as the final ML recommender.
