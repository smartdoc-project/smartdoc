from fastapi import FastAPI
app = FastAPI(title = "SmartDoc API")

@app.get("/health")
def health():
    return {"status": "ok"}

API_VERSION = "0.1.0"
MODEL_VERSION = "mock"

@app.get("/version")
def version():
    return {"api_version": API_VERSION, "models": {"ae": MODEL_VERSION}}

from schemas import PredictRequest, PredictResponse

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return PredictResponse(
        request_id=req.request_id,
        model=req.model,
        model_version=MODEL_VERSION,
        restored_image_base64=req.image_base64,  
        inference_ms=0.0,
        processing_ms=0.0,
    )