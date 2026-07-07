from pathlib import Path
import logging

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_core.documents import Document

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEC_USER_NAME, SEC_USER_EMAIL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_10k(ticker: str, limit: int = 1) -> None:

    # Download latest 10-K filings for a ticker.

    if not SEC_USER_NAME or not SEC_USER_EMAIL:
        raise ValueError("Please set SEC_USER_NAME and SEC_USER_EMAIL in your .env file.")

    downloader = Downloader(
        company_name=SEC_USER_NAME,
        email_address=SEC_USER_EMAIL,
        download_folder=str(RAW_DATA_DIR),
    )

    logger.info("Downloading %s 10-K filing(s) for %s", limit, ticker)
    downloader.get("10-K", ticker, limit=limit)


def find_downloaded_filing(ticker: str) -> Path:

    # Find the downloaded primary filing HTML file.

    ticker_dir = RAW_DATA_DIR / "sec-edgar-filings" / ticker / "10-K"

    html_files = list(ticker_dir.rglob("*.htm"))
    html_files += list(ticker_dir.rglob("*.html"))

    if html_files:
        return max(html_files, key=lambda x: x.stat().st_size)

    txt_files = list(ticker_dir.rglob("*.txt"))

    if txt_files:
        return max(txt_files, key=lambda x: x.stat().st_size)

    raise FileNotFoundError(
        f"No filing found for ticker {ticker}"
    )


def clean_filing(file_path: Path) -> str:

    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    html_start = raw_text.lower().find("<html")
    if html_start != -1:
        raw_text = raw_text[html_start:]

    soup = BeautifulSoup(raw_text, "lxml")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "ix:header",
        "ix:hidden",
        "xbrli:context",
        "xbrli:unit",
    ]):
        tag.decompose()

    for tag in soup.find_all(style=True):
        style = tag.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()

    text = soup.get_text(separator="\n")

    lines = []
    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        noisy_prefixes = (
            "dei:",
            "ix:",
            "xbrli:",
            "xbrldi:",
            "us-gaap:",
            "iso4217:",
            "http://",
            "https://",
        )

        if line.lower().startswith(noisy_prefixes):
            continue

        if len(line) < 2:
            continue

        lines.append(line)

    return "\n".join(lines)


def save_processed_text(ticker: str, text: str) -> Path:

    # Save cleaned filing text.

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / f"{ticker}_10k.txt"
    output_path.write_text(text, encoding="utf-8")

    logger.info("Saved cleaned text to %s", output_path)

    return output_path


def load_processed_document(ticker: str) -> Document:

    # Load cleaned filing as a LangChain Document.
    file_path = PROCESSED_DATA_DIR / f"{ticker}_10k.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    return Document(
        page_content=text,
        metadata={
            "ticker": ticker,
            "source": str(file_path),
            "filing_type": "10-K",
        },
    )


def ingest_company(ticker: str, limit: int = 1) -> Document:
    # Full ingestion flow:
    download_10k(ticker=ticker, limit=limit)

    filing_path = find_downloaded_filing(ticker)
    cleaned_text = clean_filing(filing_path)
    save_processed_text(ticker, cleaned_text)

    return load_processed_document(ticker)


if __name__ == "__main__":
    doc = ingest_company("JPM", limit=1)

    print("Document loaded successfully.")
    print("Characters:", len(doc.page_content))
    print("Metadata:", doc.metadata)