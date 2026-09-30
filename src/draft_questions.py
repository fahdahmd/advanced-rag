import argparse
import json
import random
import re

from tqdm import tqdm

from src.config import ROOT
from src.corpus import full_text, load_filings, normalize
from src.llm import chat

OUT_PATH = ROOT / "eval" / "drafts.jsonl"

# Edit if a cover page says otherwise
PERIOD = {"AAPL": "fiscal 2025", "MSFT": "fiscal 2026", "NVDA": "fiscal 2026", "TSLA": "2025"}
ITEMS = {"1", "1A", "7", "8", "15"}
WINDOW = 1800
CATEGORIES = {"figure", "table", "term", "text"}

PROMPT = """Below is a passage from {company}'s 10-K ({period}). Write ONE test question about it.

Rules:
- Name the company and period in the question.
- Do not copy the passage's wording. Paraphrase like a real user would.
- The answer must be short (a number with its unit, a name, or one sentence).
- "evidence" must be ONE quote copied exactly, character for character, from the passage:
  the shortest piece that proves the answer (a table row is fine).
- "category": figure (one number), table (a detail from a table), term (an exact name),
  or text (a description or explanation).
- If the passage is boilerplate with no clear factual content, reply {{"skip": true}}.

Reply with JSON only:
{{"category": "...", "question": "...", "answer": "...", "evidence": "..."}}

Passage:
{passage}"""


def sample_window(rng: random.Random, text: str) -> str:
    if len(text) <= WINDOW:
        return text
    start = rng.randrange(0, len(text) - WINDOW)
    return text[start : start + WINDOW]


def parse_json(reply: str) -> dict | None:
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="how many passages to try")
    parser.add_argument("--seed", type=int, default=1, help="use a new seed for a new batch")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    filings = load_filings()
    texts = {t: normalize(full_text(f)) for t, f in filings.items()}
    pools = {
        t: [s for s in f["sections"] if s["item"] in ITEMS and len(s["text"]) > 500]
        for t, f in filings.items()
    }
    tickers = list(filings)

    drafts, rejected = [], 0
    for i in tqdm(range(args.n)):
        ticker = tickers[i % len(tickers)]  # round-robin so every company is covered
        passage = sample_window(rng, rng.choice(pools[ticker])["text"])
        prompt = PROMPT.format(
            company=filings[ticker]["company"], period=PERIOD.get(ticker, ""), passage=passage
        )
        try:
            data = parse_json(chat([{"role": "user", "content": prompt}]))
        except RuntimeError:
            rejected += 1
            continue
        if not data or data.get("skip"):
            rejected += 1
            continue

        quote = str(data.get("evidence", "")).strip()
        # the quote must really exist in the filing, or the draft is thrown away
        if len(quote) < 15 or normalize(quote) not in texts[ticker] or data.get("category") not in CATEGORIES:
            rejected += 1
            continue

        n = len(drafts) + 1
        drafts.append(
            {
                "id": f"g{args.seed}-{n:02d}",
                "split": "test" if n % 3 == 0 else "dev",  # about one in three is held out
                "category": data["category"],
                "question": str(data["question"]).strip(),
                "answer": str(data["answer"]).strip(),
                "evidence": [{"ticker": ticker, "quote": quote}],
                "absent_terms": [],
            }
        )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(drafts)} drafts saved to {OUT_PATH} ({rejected} rejected)")


if __name__ == "__main__":
    main()