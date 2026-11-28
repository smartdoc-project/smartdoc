import base64
import io
from fastapi.testclient import TestClient
from PIL import Image
from main import app

client = TestClient(app)

def image_base64():
    buffer = io.BytesIO()
    Image.new("L", (64, 64), 200).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def test_predict_succes():
    reponse = client.post("/predict", json={
        "request_id": "doc-001", "model": "ae", "image_base64": image_base64()})
    assert reponse.status_code == 200
    assert reponse.json()["model_version"] == "mock"