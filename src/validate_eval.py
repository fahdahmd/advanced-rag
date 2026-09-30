import json
import sys
from collections import Counter

from src.config import ROOT
from src.corpus import full_text, load_filings, normalize

EVAL_PATH = ROOT / "eval" / "questions.jsonl"
CATEGORIES = {"figure", "term", "text", "table", "multi", "unanswerable"}
REQUIRED = {"id", "split", "category", "question", "answer", "evidence", "absent_terms"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not EVAL_PATH.exists():
        print(f"No question file yet. Create {EVAL_PATH} with one JSON object per line.")
        sys.exit(1)
    filings = load_filings()
    texts = {t: normalize(full_text(f)) for t, f in filings.items()}

    errors, rows, seen_ids = [], [], set()
    for n, line in enumerate(EVAL_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")
            continue
        rows.append(q)
        qid = q.get("id", f"line {n}")

        missing = REQUIRED - q.keys()
        if missing:
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: unknown category '{q['category']}'")
        if q["split"] not in {"dev", "test"}:
            errors.append(f"{qid}: split must be 'dev' or 'test'")
        if "FILL" in json.dumps(q):
            errors.append(f"{qid}: still has a FILL placeholder")

        if q["category"] == "unanswerable":
            if q["answer"] != "NOT_FOUND" or q["evidence"]:
                errors.append(f"{qid}: unanswerable needs answer NOT_FOUND and no evidence")
            if not q["absent_terms"]:
                errors.append(f"{qid}: unanswerable needs absent_terms")
            for term in q["absent_terms"]:
                for ticker, text in texts.items():
                    if normalize(term) in text:
                        errors.append(f"{qid}: '{term}' DOES appear in {ticker}, so it's answerable")
            continue

        if not q["evidence"]:
            errors.append(f"{qid}: needs at least one evidence quote")
        for ev in q["evidence"]:
            ticker = ev.get("ticker", "")
            if ticker not in texts:
                errors.append(f"{qid}: unknown ticker '{ticker}'")
            elif normalize(ev.get("quote", "")) not in texts[ticker]:
                errors.append(f"{qid}: quote NOT found in {ticker}: {ev.get('quote', '')[:60]!r}")
        if q["category"] == "multi" and len({e.get("ticker") for e in q["evidence"]}) < 2:
            errors.append(f"{qid}: 'multi' needs evidence from at least 2 companies")

    print(f"{len(rows)} questions")
    print("By category:", dict(Counter(r.get("category") for r in rows)))
    print("By split:   ", dict(Counter(r.get("split") for r in rows)))
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("\nAll good.")


if __name__ == "__main__":
    main()