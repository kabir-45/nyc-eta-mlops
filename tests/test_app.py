from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from app.server import app

client = TestClient(app)

def test_app_startup():
    response = client.get("/docs")
    assert response.status_code == 200