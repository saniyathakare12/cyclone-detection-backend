from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, log_prediction, get_all_predictions
from inference import load_cyclone_model, predict_cyclone_path
from pydantic import BaseModel
from alert import generate_alert, check_risk, CONFIDENCE_RADIUS_KM_BY_STEP

app = FastAPI()
lstm_model = None
lstm_checkpoint = None
class PathRequest(BaseModel):
    readings: list[list[float]]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    global lstm_model, lstm_checkpoint
    lstm_model, lstm_checkpoint = load_cyclone_model("best_lstm_early.pt")


@app.get("/")
def home():
    return {"message": "Cyclone backend is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")

    log_prediction(file.filename)

    return {
        "filename": file.filename,
        "cyclone_detected": True,
        "message": "Cyclone Detected"
    }
@app.post("/predict-path")
async def predict_path(request: PathRequest):
    try:
        predictions = predict_cyclone_path(lstm_model, lstm_checkpoint, request.readings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    alerts = []
    for step, pred in enumerate(predictions, start=1):
        radius_km = CONFIDENCE_RADIUS_KM_BY_STEP.get(step, 90)
        at_risk = check_risk(pred["lat"], pred["lon"], radius_km)
        if at_risk:
            alerts.append({
                "hours_ahead": pred["hours_ahead"],
                "cities_at_risk": at_risk,
                "message": generate_alert(pred, step),
            })

    return {
        "predictions": predictions,
        "alert_triggered": len(alerts) > 0,
        "alerts": alerts,
    }
@app.get("/history")
def history():
    rows = get_all_predictions()
    return [
        {"id": r[0], "timestamp": r[1], "filename": r[2]}
        for r in rows
    ]