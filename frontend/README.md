# React Frontend

This directory contains the React/Vite user interface for the survival-dataset-only rebuild.

## Development

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` when the FastAPI backend is hosted separately:

```text
VITE_API_URL=http://127.0.0.1:8000/api/v1/recommendations
```

The frontend expects the FastAPI response shape:

```json
{
  "success": true,
  "recommendations": []
}
```

## Production

```powershell
npm run build
```

The generated `dist/` directory can be served by a web server. The frontend calls the FastAPI endpoint configured through `VITE_API_URL`.

## Current prototype boundary

The current form collects a basic site profile, but the backend currently summarizes `data/survival_ml_final.csv`. Climate, rainfall, and longitude/latitude-derived features will be added in a later rebuild phase.
