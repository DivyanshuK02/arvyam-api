# tests/test_packaging_tiers.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root


def test_catalog_packaging_mapping_and_lg_rule():
    cat_path = ROOT / "app" / "catalog.json"
    assert cat_path.exists(), f"missing {cat_path}"
    data = json.loads(cat_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) > 0, "catalog.json must be a non-empty list"

    allowed_tiers = {"Classic", "Signature", "Luxury"}
    allowed_packaging = {"Box", "Vase", "PremiumBox"}

    for row in data:
        # minimal shape sanity
        assert isinstance(row, dict) and "id" in row, "each catalog row must be an object with id"
        tier = row.get("tier")
        pkg = row.get("packaging")

        # tier + packaging enums must be valid
        assert tier in allowed_tiers, f"invalid tier for {row.get('id')}: {tier}"
        assert pkg in allowed_packaging, f"invalid packaging for {row.get('id')}: {pkg}"

        # frozen mapping by tier
        if tier == "Classic":
            assert pkg == "Box", f"{row['id']}: Classic must map to Box"
        elif tier == "Signature":
            assert pkg == "Vase", f"{row['id']}: Signature must map to Vase"
        elif tier == "Luxury":
            assert pkg == "PremiumBox", f"{row['id']}: Luxury must map to PremiumBox"

        # LG rule: if true, tier must be Luxury (missing/False is fine)
        lg = bool(row.get("luxury_grand", False))
        if lg:
            assert tier == "Luxury", f"{row['id']}: luxury_grand=true only allowed inside Luxury"

        # ban keepsake variants at the data level
        assert "Keepsake" not in str(pkg), f"{row['id']}: keepsake packaging not allowed"
        assert pkg != "Premium Box", f"{row['id']}: packaging enum must be 'PremiumBox' (no space)"
