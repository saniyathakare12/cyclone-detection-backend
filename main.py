from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, log_prediction, get_all_predictions
from model import load_model, run_inference

app = FastAPI()

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
    load_model()


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

    category, confidence = run_inference(contents)

    log_prediction(file.filename, category, confidence)

    return {
        "filename": file.filename,
        "cyclone_detected": True,
        "category": category,
        "confidence": confidence
    }


@app.get("/history")
def history():
    rows = get_all_predictions()
    return [
        {"id": r[0], "timestamp": r[1], "filename": r[2], "category": r[3], "confidence": r[4]}
        for r in rows
    ]