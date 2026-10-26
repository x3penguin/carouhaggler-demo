# Data Documentation

## 1. Singapore Marketplace Listings (`listings_sg.json`)
- **Count:** 104 verified listings across 8 distinct categories (phones, laptops, gaming, audio, furniture, sneakers, textbooks, appliances).
- **Format:** Each listing contains:
  - `id`: Unique identifier (e.g. `phone-001`, `laptop-012`).
  - `title`: Item model, variant, specifications, and color.
  - `category`: Category string.
  - `description`: Localized Singapore marketplace context (MRT meetups, HDB clearances, reason for selling).
  - `condition_facts`: Ground-truth condition facts, containing at least one explicit flaw (e.g. battery health, dents, missing accessories, worn soles, water rings) for RQ4 honesty evaluations.
  - `asking_price`: Realistic SGD market price.

## 2. Sold Price Comparables Database (`comparables.json`)
- **Count:** 416 price records.
- **Sources (Non-Marketplace):**
  - Apple Certified Refurbished SG Official Store
  - Samsung Official Trade-In Valuation Matrix SG
  - Reebelo Singapore Certified Pre-Owned Index
  - Challenger Tech Singapore Clearance Catalog
  - Harvey Norman SG Ex-Display Floor Clearance
  - Courts Singapore Floor Demo & Renewed Stock
  - IKEA As-Is Circular Hub SG Benchmark
  - Dyson Official Renewed Outlet SG
  - SMU/NUS Student Book Exchange Guild Pricing
  - PlayStation Official Factory Refurbished Program
  - Novelship / KicksCrew Verified Secondary Trade SG

## 3. Data Validation
To re-validate the schema and coverage:
```bash
uv run python data/validate_comparables.py
```
To generate or re-compile the dataset:
```bash
uv run python data/build_dataset.py
```
