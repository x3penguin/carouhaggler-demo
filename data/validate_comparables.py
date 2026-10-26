"""Validate (and optionally convert) data/comparables.json.

    uv run python data/validate_comparables.py                 # check data/comparables.json
    uv run python data/validate_comparables.py --csv in.csv    # convert CSV -> comparables.json, then check

Errors (exit code 1) are schema problems. Warnings are coverage gaps: a listing with fewer
than --min-per-item matching comparables, which makes price_lookup medians unreliable.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from haggle.scenarios import load_listings
from haggle.tools.price_lookup import price_lookup

ROOT = Path(__file__).resolve().parent
CONDITIONS = {"like_new", "good", "fair", "poor"}
FIELDS = ["id", "category", "title", "condition", "price_sgd", "age_months", "source", "notes"]


def convert_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            age = (r.get("age_months") or "").strip()
            rows.append(
                {
                    "id": r["id"].strip(),
                    "category": r["category"].strip(),
                    "title": r["title"].strip(),
                    "condition": r["condition"].strip(),
                    "price_sgd": float(r["price_sgd"]),
                    "age_months": int(age) if age else None,
                    "source": r["source"].strip(),
                    "notes": (r.get("notes") or "").strip(),
                }
            )
    return rows


def check_schema(entries: list[dict], categories: set[str]) -> list[str]:
    errors, seen = [], set()
    for n, e in enumerate(entries):
        where = f"entry {n} ({e.get('id', '?')})"
        missing = [k for k in FIELDS if k not in e]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue
        if e["id"] in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(e["id"])
        if e["category"] not in categories:
            errors.append(f"{where}: category '{e['category']}' not in listings {sorted(categories)}")
        if e["condition"] not in CONDITIONS:
            errors.append(f"{where}: condition must be one of {sorted(CONDITIONS)}")
        if not isinstance(e["price_sgd"], (int, float)) or e["price_sgd"] <= 0:
            errors.append(f"{where}: price_sgd must be a positive number")
        if e["age_months"] is not None and not isinstance(e["age_months"], int):
            errors.append(f"{where}: age_months must be an integer or null")
        if not str(e["source"]).strip():
            errors.append(f"{where}: source is required")
        elif "PLACEHOLDER" in str(e["source"]):
            errors.append(f"{where}: still a PLACEHOLDER source")
    return errors


def coverage_warnings(listings, path: Path, min_per_item: int) -> list[str]:
    warnings = []
    for lst in listings:
        n = price_lookup(lst.title, lst.category, path=path)["n"]
        if n < min_per_item:
            warnings.append(f"{lst.id} '{lst.title}': only {n} matching comparables (want >= {min_per_item})")
    return warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--min-per-item", type=int, default=8)
    ap.add_argument("--path", type=Path, default=ROOT / "comparables.json")
    args = ap.parse_args(argv)

    if args.csv:
        args.path.write_text(json.dumps(convert_csv(args.csv), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.path}")

    entries = json.loads(args.path.read_text(encoding="utf-8"))
    listings = load_listings(ROOT / "listings_sg.json")
    errors = check_schema(entries, {lst.category for lst in listings})
    warnings = coverage_warnings(listings, args.path, args.min_per_item)

    for e in errors:
        print("ERROR  ", e)
    for w in warnings:
        print("WARNING", w)
    print(f"{len(entries)} entries, {len(errors)} errors, {len(warnings)} coverage warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
