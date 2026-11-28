from typing import Literal
from pydantic import BaseModel, Field
MAX_IMAGE_B64 = 4_000_000   

class PredictRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=200)
    model: Literal["ae"] = "ae"          # un seul modèle pour l'instant
    image_base64: str = Field(min_length=1, max_length=MAX_IMAGE_B64)

class PredictResponse(BaseModel):
    request_id: str
    model: Literal["ae"]
    model_version: str
    restored_image_base64: str
    inference_ms: float
    processing_ms: float