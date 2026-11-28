import base64
import io
import time
from fastapi import FastAPI, HTTPException
from PIL import Image
from schemas import PredictRequest, PredictResponse
app = FastAPI(title = "SmartDoc API")

API_VERSION = "0.1.0"
MODEL_VERSION = "mock"
MAX_COTE = 2048 #pixels a ajuster 
stats = {"appels": 0, "temps_total_ms": 0.0}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"api_version": API_VERSION, "models": {"ae": MODEL_VERSION}}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    debut = time.perf_counter()   
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_base64, validate=True)))
        image.load()
    except Exception:
        raise HTTPException(status_code=422, detail="image invalide")

    if image.format not in ("PNG", "JPEG") or max(image.size) > MAX_COTE:
        raise HTTPException(status_code=422, detail="PNG ou JPEG de 2048 px maximum")
    image = image.convert("L")   

    t0 = time.perf_counter()
    restauree = image
    inference_ms = (time.perf_counter() - t0) * 1000

    buffer = io.BytesIO()
    restauree.save(buffer, format="PNG")
    resultat = base64.b64encode(buffer.getvalue()).decode()
    total_ms = (time.perf_counter() - debut) * 1000
    stats["appels"] += 1
    stats["temps_total_ms"] += total_ms

    return PredictResponse(
        request_id=req.request_id,
        model=req.model,
        model_version=MODEL_VERSION,
        restored_image_base64=resultat,  
        inference_ms=round(inference_ms, 2),
        processing_ms=round(total_ms, 2),
    )

@app.get("/metrics")
def metrics():
    n = stats["appels"]
    moyenne = stats["temps_total_ms"] / n if n else 0.0
    return {"calls": n, "avg_latency_ms": round(moyenne, 2)}
