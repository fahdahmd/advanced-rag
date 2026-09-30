import re
import sys

from src.corpus import load_filings


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print('Usage: python -m src.find TICKER "text to search"')
        return

    ticker = sys.argv[1].upper()
    needle = " ".join(sys.argv[2:])
    filings = load_filings()
    if ticker not in filings:
        print(f"Unknown ticker. Available: {list(filings)}")
        return

    shown = 0
    for section in filings[ticker]["sections"]:
        for m in re.finditer(re.escape(needle), section["text"], re.IGNORECASE):
            start = max(0, m.start() - 150)
            context = section["text"][start : m.end() + 250].replace("\n", " / ")
            print(f"[Item {section['item']}] ...{context}...\n")
            shown += 1
            if shown >= 8:
                print("(showing first 8 matches)")
                return
    if shown == 0:
        print("No matches.")


if __name__ == "__main__":
    main()