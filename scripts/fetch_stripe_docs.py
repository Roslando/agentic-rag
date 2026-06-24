"""
Fetch Stripe documentation pages as Markdown and save them locally.

Usage:
    # Fetch from the provided index file (recommended)
    python scripts/fetch_stripe_docs.py --index-file data/stripe_docs_index.md

    # Fetch from Stripe's sitemap (gets ALL docs, ~1000+ pages)
    python scripts/fetch_stripe_docs.py --from-sitemap

    # Dry-run: only show which URLs would be fetched
    python scripts/fetch_stripe_docs.py --index-file data/stripe_docs_index.md --dry-run

How to prepare the index file:
    Save the Stripe documentation index (the markdown text with all the doc links)
    into:  data/stripe_docs_index.md
    Then run this script.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STRIPE_BASE = "https://docs.stripe.com"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "markdown_docs" / "stripe"

HEADERS = {
    "User-Agent": "RAG-Agent/1.0 (educational; stripe-docs-indexer)",
    "Accept": "text/markdown, text/plain, */*",
}

# Map URL path prefix → human-readable category
CATEGORY_MAP = {
    "payments": "Payments",
    "billing": "Billing",
    "connect": "Connect",
    "checkout": "Checkout",
    "api": "API Reference",
    "webhooks": "Webhooks",
    "radar": "Radar / Fraud",
    "terminal": "Terminal",
    "tax": "Tax",
    "issuing": "Issuing",
    "treasury": "Treasury",
    "identity": "Identity",
    "capital": "Capital",
    "crypto": "Crypto",
    "revenue-recognition": "Revenue Recognition",
    "financial-connections": "Financial Connections",
    "sigma": "Sigma",
    "invoicing": "Invoicing",
    "climate": "Climate",
    "atlas": "Atlas",
    "get-started": "Get Started",
    "no-code": "No Code",
    "products-prices": "Products & Prices",
    "elements": "Elements",
    "payment-links": "Payment Links",
    "declines": "Declines",
    "disputes": "Disputes",
    "security": "Security",
}


def _infer_category(url: str) -> str:
    path = urlparse(url).path.strip("/")
    first_segment = path.split("/")[0]
    return CATEGORY_MAP.get(first_segment, "General")


def _url_to_slug(url: str) -> str:
    """Convert a Stripe docs URL to a safe filename slug."""
    path = urlparse(url).path.strip("/")
    # Remove .md suffix, replace slashes with underscores
    slug = path.replace(".md", "").replace("/", "_")
    # Remove trailing/leading underscores, collapse multiples
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "index"


def extract_urls_from_markdown(text: str) -> list[str]:
    """Extract all unique Stripe docs URLs from a Markdown document."""
    # Match URLs in markdown links: (https://docs.stripe.com/...)
    pattern = r'https://docs\.stripe\.com/[^\s\)\"\'\>]+'
    raw = re.findall(pattern, text)

    urls = set()
    for url in raw:
        # Strip trailing punctuation
        url = url.rstrip(".,;:")
        # Ensure .md suffix
        if not url.endswith(".md"):
            url = url + ".md"
        # Normalize: remove fragment (#...) and query (?...)
        base = url.split("?")[0].split("#")[0]
        urls.add(base)

    sorted_urls = sorted(urls)
    logger.info(f"Found {len(sorted_urls)} unique URLs in index")
    return sorted_urls


def fetch_from_sitemap() -> list[str]:
    """Fetch all Stripe doc URLs from their sitemap XML."""
    sitemap_url = f"{STRIPE_BASE}/sitemap.xml"
    logger.info(f"Fetching sitemap: {sitemap_url}")
    resp = requests.get(sitemap_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Extract <loc> entries
    locs = re.findall(r"<loc>([^<]+)</loc>", resp.text)
    urls = []
    for loc in locs:
        if "docs.stripe.com" in loc:
            md_url = loc.rstrip("/") + ".md" if not loc.endswith(".md") else loc
            urls.append(md_url)

    logger.info(f"Sitemap: {len(urls)} URLs")
    return sorted(set(urls))


def fetch_page(url: str) -> str | None:
    """Fetch a single Stripe doc page as Markdown. Returns content or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            content = resp.text
            # Stripe returns HTML for some URLs even with .md — skip those
            if content.strip().startswith("<!DOCTYPE") or "<html" in content[:200]:
                logger.debug(f"HTML response (skipped): {url}")
                return None
            return content
        elif resp.status_code == 404:
            logger.debug(f"404 not found: {url}")
            return None
        else:
            logger.warning(f"HTTP {resp.status_code}: {url}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Request failed ({e}): {url}")
        return None


def save_page(url: str, content: str, output_dir: Path) -> Path:
    """Save Markdown content and companion metadata to disk."""
    slug = _url_to_slug(url)
    md_path = output_dir / f"{slug}.md"
    meta_path = output_dir / f"{slug}.meta.json"

    md_path.write_text(content, encoding="utf-8")

    meta = {
        "url": url,
        "category": _infer_category(url),
        "slug": slug,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path


def run(
    urls: list[str],
    output_dir: Path,
    delay: float = 0.5,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Download all URLs. Returns (downloaded, failed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded, failed = 0, 0

    for i, url in enumerate(urls, 1):
        slug = _url_to_slug(url)
        md_path = output_dir / f"{slug}.md"

        # Skip already downloaded
        if md_path.exists():
            logger.debug(f"[{i}/{len(urls)}] Skipped (exists): {slug}")
            downloaded += 1
            continue

        if dry_run:
            logger.info(f"[DRY] {url} → {slug}.md")
            continue

        logger.info(f"[{i}/{len(urls)}] Fetching: {url}")
        content = fetch_page(url)

        if content:
            save_page(url, content, output_dir)
            downloaded += 1
            logger.info(f"  OK Saved {slug}.md ({len(content):,} chars)")
        else:
            failed += 1

        time.sleep(delay)

    return downloaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Stripe docs as Markdown")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--index-file",
        type=Path,
        default=Path("data/stripe_docs_index.md"),
        help="Markdown file containing the Stripe docs index with links",
    )
    group.add_argument(
        "--from-sitemap",
        action="store_true",
        help="Fetch all URLs from Stripe's sitemap.xml instead",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without fetching")
    args = parser.parse_args()

    if args.from_sitemap:
        urls = fetch_from_sitemap()
    else:
        if not args.index_file.exists():
            logger.error(
                f"Index file not found: {args.index_file}\n"
                f"Save the Stripe documentation index to that path and retry.\n"
                f"(Copy the Stripe docs index markdown content into data/stripe_docs_index.md)"
            )
            sys.exit(1)
        text = args.index_file.read_text(encoding="utf-8")
        urls = extract_urls_from_markdown(text)

    if not urls:
        logger.error("No URLs found. Check your index file.")
        sys.exit(1)

    logger.info(f"\nTarget: {args.output_dir}")
    logger.info(f"URLs to process: {len(urls)}")
    logger.info(f"Delay: {args.delay}s between requests\n")

    downloaded, failed = run(urls, args.output_dir, args.delay, args.dry_run)

    print(
        f"\n{'[DRY RUN] ' if args.dry_run else ''}Results:\n"
        f"  [OK]   Downloaded / already cached : {downloaded}\n"
        f"  [FAIL] Failed / skipped            : {failed}\n"
        f"  Output directory                   : {args.output_dir}\n"
    )


if __name__ == "__main__":
    main()
