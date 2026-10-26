"""price_lookup: retrieve comparable sold prices for an item (deterministic, no LLM).

Pure-Python BM25 over the titles in data/comparables.json, so there is no new dependency and
results are reproducible: ties are broken by comparable id, never by iteration order.
"""

import json
import math
import re
import statistics
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "comparables.json"
_TOKEN = re.compile(r"[a-z0-9]+")
K1, B = 1.5, 0.75
MAX_MATCHES_FOR_MEDIAN = 10


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _Index:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.docs = [tokenize(e["title"]) for e in entries]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs) if self.docs else 0.0
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        doc = self.docs[i]
        total = 0.0
        for t in set(query_tokens):
            tf = doc.count(t)
            if tf:
                norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(doc) / self.avg_len))
                total += self.idf[t] * norm
        return total


@lru_cache(maxsize=8)
def _index(path: str) -> _Index:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return _Index(entries)


def price_lookup(
    query: str, category: str | None = None, k: int = 3, path: str | Path = DEFAULT_PATH
) -> dict:
    """Median comparable price plus the k closest examples.

    Returns {"median": float | None, "n": int, "examples": [...]}. n is how many comparables
    matched; the median uses the best MAX_MATCHES_FOR_MEDIAN of them.
    """
    index = _index(str(path))
    q = tokenize(query)
    scored = []
    for i, entry in enumerate(index.entries):
        if category is not None and entry["category"] != category:
            continue
        s = index.score(q, i)
        if s > 0:
            scored.append((-s, entry["id"], entry))
    scored.sort(key=lambda x: (x[0], x[1]))
    matches = [e for _, _, e in scored]
    if not matches:
        return {"median": None, "n": 0, "examples": []}
    median = statistics.median(e["price_sgd"] for e in matches[:MAX_MATCHES_FOR_MEDIAN])
    keep = ("id", "title", "condition", "price_sgd", "age_months")
    return {
        "median": float(median),
        "n": len(matches),
        "examples": [{key: e.get(key) for key in keep} for e in matches[:k]],
    }
