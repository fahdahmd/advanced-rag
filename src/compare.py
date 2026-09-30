import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["results"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3:
        print("Usage: python -m src.compare OLD_RESULTS.json NEW_RESULTS.json")
        return
    old, new = load(sys.argv[1]), load(sys.argv[2])

    gained, lost = [], []
    for qid, n in new.items():
        o = old.get(qid)
        if not o or o["recall"] is None or n["recall"] is None:
            continue
        if n["recall"] > o["recall"]:
            gained.append(n)
        elif n["recall"] < o["recall"]:
            lost.append(n)

    print(f"Gained recall on {len(gained)} question(s), lost on {len(lost)}\n")
    for label, rows in (("GAINED", gained), ("LOST", lost)):
        for r in rows:
            print(f"{label} [{r['category']}] {r['id']}: {r['question']}")


if __name__ == "__main__":
    main()