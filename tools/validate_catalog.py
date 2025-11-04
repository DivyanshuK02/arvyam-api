#!/usr/bin/env python3
"""
Validate app/catalog.json against docs/catalog.schema.json (Draft 2020-12).
Usage:
  python tools/validate_catalog.py app/catalog.json docs/catalog.schema.json
"""
import sys
import json
from jsonschema import validate, Draft202012Validator

def main(cat_path: str, schema_path: str) -> None:
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    with open(cat_path, encoding="utf-8") as f:
        data = json.load(f)
    # Validate the schema itself, then the data
    Draft202012Validator.check_schema(schema)
    validate(instance=data, schema=schema)
    print("[catalog.schema] OK")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: validate_catalog.py <catalog.json> <schema.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
