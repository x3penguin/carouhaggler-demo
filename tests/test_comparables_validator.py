import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("validate_comparables", Path("data/validate_comparables.py"))
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)

GOOD = dict(id="x", category="phones", title="iPhone 13", condition="good", price_sgd=500, age_months=12, source="retail page", notes="")


def test_valid_entry_passes():
    assert vc.check_schema([GOOD], {"phones"}) == []


def test_bad_entries_are_reported():
    bad = [
        {**GOOD, "id": "a", "condition": "mint"},
        {**GOOD, "id": "b", "price_sgd": -1},
        {**GOOD, "id": "c", "category": "boats"},
        {**GOOD, "id": "d", "source": ""},
        {**GOOD, "id": "e", "source": "PLACEHOLDER - replace"},
        {**GOOD, "id": "a"},
    ]
    errors = vc.check_schema(bad, {"phones"})
    assert len(errors) == 6
    assert any("duplicate" in e for e in errors)


def test_missing_fields_reported():
    assert "missing fields" in vc.check_schema([{"id": "z"}], {"phones"})[0]


def test_csv_conversion(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("id,category,title,condition,price_sgd,age_months,source,notes\nz,phones,iPhone,good,500,,shop,\n")
    rows = vc.convert_csv(f)
    assert rows[0]["price_sgd"] == 500.0 and rows[0]["age_months"] is None
    assert vc.check_schema(rows, {"phones"}) == []


def test_main_flags_placeholder_data(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{**GOOD, "source": "PLACEHOLDER x"}]))
    assert vc.main(["--path", str(p)]) == 1
