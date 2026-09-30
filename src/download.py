import json
import time

import httpx

from src.config import settings

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
HEADERS = {"User-Agent": settings.sec_user_agent}


def get(url: str) -> httpx.Response:
    time.sleep(0.2)  # SEC allows at most 10 requests per second
    resp = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return resp


def load_cik_map() -> dict[str, int]:
    """SEC identifies companies by CIK number, not ticker."""
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"]: row["cik_str"] for row in data.values()}


def find_latest_10k(cik: int) -> dict:
    data = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):  # list is newest-first
        if form == "10-K":
            accession = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            return {
                "company": data["name"],
                "cik": cik,
                "filing_date": recent["filingDate"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}",
            }
    raise ValueError(f"No 10-K found for CIK {cik}")


def main():
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_map()
    for ticker in TICKERS:
        html_path = settings.raw_dir / f"{ticker}.html"
        if html_path.exists():
            print(f"{ticker}: already downloaded, skipping")
            continue
        info = find_latest_10k(cik_map[ticker])
        info["ticker"] = ticker
        html_path.write_text(get(info["url"]).text, encoding="utf-8")
        (settings.raw_dir / f"{ticker}.json").write_text(json.dumps(info, indent=2))
        print(f"{ticker}: saved {info['company']} 10-K filed {info['filing_date']}")


if __name__ == "__main__":
    main()