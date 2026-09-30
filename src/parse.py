import json
import re

from bs4 import BeautifulSoup

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
    """Split by 'Item X'. The table of contents also matches, so for each item
    we keep the occurrence with the longest text (the real section)."""
    matches = list(ITEM_PATTERN.finditer(text))
    best: dict[str, dict] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        item = m.group(1).upper()
        if item not in best or len(body) > len(best[item]["text"]):
            best[item] = {
                "item": item,
                "title": ITEM_NAMES.get(item, ""),
                "start": m.start(),
                "text": body,
            }
    return sorted(best.values(), key=lambda s: s["start"])


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
        print(f"{ticker}: {len(text):,} chars, items: {[s['item'] for s in sections]}")


if __name__ == "__main__":
    main()