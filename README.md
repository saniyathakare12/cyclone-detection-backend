# Cyclone Detection Backend

FastAPI backend for the cyclone detection & intensity classification project. Accepts a satellite image, returns a predicted cyclone category and confidence score, and logs predictions to a local database.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000` — interactive docs at `/docs`.

## Endpoints

- `POST /predict` — upload an image, get back `{filename, cyclone_detected, category, confidence}`. Currently returns placeholder predictions until the trained model is integrated.
- `GET /history` — returns all logged predictions.

## Status

Model integration pending — currently using randomized placeholder predictions so the frontend can be built against a stable API contract.