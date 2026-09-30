import json
import sys

from src.config import ROOT

DRAFTS = ROOT / "eval" / "drafts.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if QUESTIONS.exists():
        text = QUESTIONS.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            QUESTIONS.write_text(text + "\n", encoding="utf-8")

    drafts, existing = read(DRAFTS), read(QUESTIONS)
    remaining = list(drafts)
    for d in drafts:
        ev = d["evidence"][0]
        print("\n" + "-" * 70)
        print(f"[{d['category']}] {d['question']}")
        print(f"Answer:   {d['answer']}")
        print(f"Evidence ({ev['ticker']}): {ev['quote']}")
        choice = input("y = keep, Enter = skip, q = quit: ").strip().lower()
        if choice == "q":
            break
        remaining.remove(d)
        if choice == "y":
            d["id"] = f"q{len(existing) + 1:03d}"
            existing.append(d)
            with QUESTIONS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    DRAFTS.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in remaining), encoding="utf-8"
    )
    print(f"\n{len(existing)} questions in total, {len(remaining)} drafts left.")


if __name__ == "__main__":
    main()