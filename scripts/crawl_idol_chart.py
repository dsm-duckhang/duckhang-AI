from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


BASE_URL = "https://www.idol-chart.com"
RANKING_URL = f"{BASE_URL}/ranking/"
USER_AGENT = "duckhang-AI ranking collector/1.0"


class RankingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, Any]] = []
        self.period = ""
        self._div_depth = 0
        self._top_days_depth: int | None = None
        self._item_depth: int | None = None
        self._item: dict[str, Any] | None = None
        self._capture: str | None = None
        self._capture_tag: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "div":
            self._div_depth += 1
            if "top_days" in classes:
                self._top_days_depth = self._div_depth
            if "idol_item" in classes and self._item is None:
                self._item = {}
                self._item_depth = self._div_depth

        if self._item is not None:
            if "no" in classes:
                self._start_capture("rank", tag)
            elif "txt01" in classes:
                self._start_capture("name", tag)

            if tag == "img" and not self._item.get("image_url"):
                src = attributes.get("src")
                if src:
                    self._item["image_url"] = urljoin(BASE_URL, src)

            if tag == "a" and not self._item.get("profile_url"):
                href = attributes.get("href")
                if href and href.startswith("/profile/"):
                    self._item["profile_url"] = urljoin(BASE_URL, href)
        elif "tit" in classes and self._top_days_depth is not None:
            self._start_capture("period", tag)

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture and tag == self._capture_tag:
            value = " ".join("".join(self._text).split())
            if self._capture == "period":
                self.period = value
            elif self._item is not None:
                self._item[self._capture] = value
            self._capture = None
            self._capture_tag = None
            self._text = []

        if tag == "div":
            if self._item is not None and self._item_depth == self._div_depth:
                if {"rank", "name", "image_url"}.issubset(self._item):
                    self._item["rank"] = int(self._item["rank"])
                    self.items.append(self._item)
                self._item = None
                self._item_depth = None
            if self._top_days_depth == self._div_depth:
                self._top_days_depth = None
            self._div_depth -= 1

    def _start_capture(self, field: str, tag: str) -> None:
        self._capture = field
        self._capture_tag = tag
        self._text = []


@dataclass
class RateLimiter:
    delay: float
    last_request_at: float | None = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            remaining = self.delay - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


def fetch(url: str, limiter: RateLimiter) -> tuple[bytes, str]:
    limiter.wait()
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read(), response.headers.get_content_type()


def parse_page(html: bytes) -> RankingParser:
    parser = RankingParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    return parser


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name).strip("._")
    return cleaned or "artist"


def image_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return suffix
    return mimetypes.guess_extension(content_type) or ".jpg"


def write_results(output_dir: Path, period: str, items: list[dict[str, Any]]) -> None:
    payload = {
        "source": RANKING_URL,
        "ranking_type": "monthly",
        "period": period,
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "count": len(items),
        "artists": items,
    }
    (output_dir / "monthly_top_50.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = ["rank", "name", "image_url", "image_file", "profile_url"]
    with (output_dir / "monthly_top_50.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)


def crawl(limit: int, output_dir: Path, delay: float, overwrite: bool) -> list[dict[str, Any]]:
    limiter = RateLimiter(delay)
    artists: list[dict[str, Any]] = []
    period = ""
    page = 1

    while len(artists) < limit:
        url = f"{RANKING_URL}?d_type=M&page={page}"
        html, _ = fetch(url, limiter)
        parsed = parse_page(html)
        if not parsed.items:
            break
        period = period or parsed.period
        artists.extend(parsed.items)
        print(f"ranking page {page}: {len(parsed.items)} artists")
        page += 1

    artists = sorted(artists, key=lambda item: item["rank"])[:limit]
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for item in artists:
        url = item["image_url"]
        url_suffix = Path(urlparse(url).path).suffix.lower() or ".jpg"
        candidate = image_dir / f"{item['rank']:02d}_{safe_name(item['name'])}{url_suffix}"
        if candidate.exists() and not overwrite:
            item["image_file"] = candidate.relative_to(output_dir).as_posix()
            print(f"image {item['rank']:02d}/{limit}: cached")
            continue

        try:
            content, content_type = fetch(url, limiter)
            destination = candidate.with_suffix(image_extension(url, content_type))
            destination.write_bytes(content)
            item["image_file"] = destination.relative_to(output_dir).as_posix()
            print(f"image {item['rank']:02d}/{limit}: {item['name']}")
        except Exception as exc:  # Keep the remaining ranking data if one image fails.
            item["image_file"] = ""
            print(f"image {item['rank']:02d}/{limit}: failed ({exc})")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(output_dir, period, artists)
    return artists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/idol_chart"),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds between requests (the site's robots.txt requests 5 seconds)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    artists = crawl(args.limit, args.output_dir, args.delay, args.overwrite)
    if len(artists) != args.limit:
        raise SystemExit(f"expected {args.limit} artists, collected {len(artists)}")
    print(f"saved {len(artists)} artists to {args.output_dir}")


if __name__ == "__main__":
    main()
