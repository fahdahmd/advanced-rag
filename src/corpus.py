import json
import re

from src.config import settings


def load_filings() -> dict[str, dict]:
    """Load every parsed filing, keyed by ticker."""
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(settings.parsed_dir.glob("*.json"))
    }


def full_text(filing: dict) -> str:
    return "\n".join(section["text"] for section in filing["sections"])


def normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so matching ignores line breaks."""
    return re.sub(r"\s+", " ", text).strip().lower()