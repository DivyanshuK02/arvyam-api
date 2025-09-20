def test_curate_contract(client):
    r = client.post("/api/curate", json={"prompt": "I’m so sorry for your loss"})
    assert r.status_code == 200
    payload = r.json()
    items = payload["items"]
    assert isinstance(items, list) and len(items) == 3

    for it in items:
        # must-have fields
        for k in ("id", "title", "desc", "image", "price", "currency"):
            assert k in it and it[k] not in (None, "")
        # must NOT leak internal fields
        assert "image_url" not in it
        assert "price_inr" not in it
