from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_curate_contract(client):
    r = client.post("/api/curate", json={"prompt": "I’m so sorry for your loss"})
    assert r.status_code == 200
    
    # FIX: The response is the list itself, not an object containing 'items'.
    items = r.json()
    
    assert isinstance(items, list)
    assert len(items) == 3
    # Check for a few required public fields in the first item
    assert "id" in items[0]
    assert "image" in items[0]
    assert "price" in items[0]
    assert "currency" in items[0]
