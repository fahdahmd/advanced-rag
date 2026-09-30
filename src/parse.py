import json
import re

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from src.config import settings

ITEM_PATTERN = re.compile(
    r"^\s*item\s+(\d{1,2}[abc]?)\s*[\.\:\-–—]?", re.IGNORECASE | re.MULTILINE
)

ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Market Risk Disclosures",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions That Prevent Inspections",
    "10": "Directors and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Accountant Fees",
    "15": "Exhibits",
    "16": "Form 10-K Summary",
}

BLOCK_TAGS = ["div", "p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def table_to_text(table) -> str:
    """Turn an HTML table into 'cell | cell | cell' lines, so numbers stay readable."""
    lines = []
    for row in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cleaned = []
        for cell in cells:
            if not cell or cell == "$":
                continue
            if cell in {")", "%", ")%"} and cleaned:
                cleaned[-1] += cell  # glue "(" numbers and percents back together
                continue
            cleaned.append(cell)
        if cleaned:
            lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name == "ix:header"):
        tag.decompose()  # hidden XBRL metadata, not real content

    for table in soup.find_all("table"):
        text = table_to_text(table)
        table.replace_with(f"\n[TABLE]\n{text}\n[/TABLE]\n" if text else "")

    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")  # keep paragraphs on their own lines

    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[dict]:
    matches = list(ITEM_PATTERN.finditer(text))
    candidates = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        candidates.append(
            {
                "item": m.group(1).upper(),
                "start": m.start(),
                "text": text[m.end():end].strip(),
            }
        )

    # The table of contents sits before the real Item 1, so find where the
    # real body starts (Item 1 = the longest "Item 1" match) and ignore
    # everything before it.
    item1 = [c for c in candidates if c["item"] == "1"]
    body_start = max(item1, key=lambda c: len(c["text"]))["start"] if item1 else 0

    best: dict[str, dict] = {}
    for c in candidates:
        if c["start"] < body_start:
            continue
        item = c["item"]
        if item not in best or len(c["text"]) > len(best[item]["text"]):
            best[item] = {**c, "title": ITEM_NAMES.get(item, "")}
    return sorted(best.values(), key=lambda s: s["start"])

KEY_ITEMS = ["1", "1A", "7", "8", "15"]
def main():
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    for html_path in sorted(settings.raw_dir.glob("*.html")):
        ticker = html_path.stem
        meta = json.loads((settings.raw_dir / f"{ticker}.json").read_text())
        text = html_to_text(html_path.read_text(encoding="utf-8"))
        sections = split_sections(text)
        out = {**meta, "sections": sections}
        (settings.parsed_dir / f"{ticker}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        sizes = {s["item"]: len(s["text"]) for s in sections}
        key = ", ".join(f"Item {k}: {sizes.get(k, 0):,}" for k in KEY_ITEMS)
        print(f"{ticker}: {len(text):,} chars | {key}")
        print(f"  order: {[s['item'] for s in sections]}")


if __name__ == "__main__":
    main()