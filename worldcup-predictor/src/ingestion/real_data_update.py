from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion.database import get_engine, initialize_database
from src.utils.config import DATABASE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


ELO_SOURCE_URL = "https://www.international-football.net/"
USER_AGENT = "WorldCupPredictor2026/1.0 (+local data updater)"


@dataclass(frozen=True)
class UpdateResult:
    source: str
    raw_path: Path
    processed_path: Path
    rows_downloaded: int
    rows_updated: int
    updated_at: str


def fetch_text(url: str = ELO_SOURCE_URL, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_elo_rankings(html: str) -> pd.DataFrame:
    published_at = _parse_published_date(html)
    pattern = re.compile(
        r"<th[^>]*>\s*(?P<rank>\d+)\.\s*</th>.*?"
        r"<span class=\"opensans\">(?P<team>[^<]+)</span>.*?"
        r"<font[^>]*>\s*(?P<elo>\d+)\s*</font>",
        re.S,
    )
    rows = []
    seen: set[str] = set()
    for match in pattern.finditer(html):
        team = _clean_html_text(match.group("team"))
        if team in seen:
            continue
        seen.add(team)
        rows.append(
            {
                "rank": int(match.group("rank")),
                "team": team,
                "elo_rating": int(match.group("elo")),
                "published_at": published_at.isoformat(),
                "source": ELO_SOURCE_URL,
            }
        )
    if not rows:
        raise ValueError("No Elo ranking rows were parsed from the source page.")
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def update_elo_data(engine: Engine | None = None) -> UpdateResult:
    engine = engine or get_engine()
    if not DATABASE_PATH.exists():
        initialize_database()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc)
    html = fetch_text()
    raw_path = RAW_DATA_DIR / f"international_football_elo_{fetched_at:%Y%m%d}.html"
    raw_path.write_text(html, encoding="utf-8")
    rankings = parse_elo_rankings(html)
    processed_path = PROCESSED_DATA_DIR / "elo_ratings_latest.csv"
    rankings.to_csv(processed_path, index=False)
    rows_updated = apply_elo_rankings(rankings, engine)
    return UpdateResult(
        source=ELO_SOURCE_URL,
        raw_path=raw_path,
        processed_path=processed_path,
        rows_downloaded=len(rankings),
        rows_updated=rows_updated,
        updated_at=fetched_at.isoformat(),
    )


def apply_elo_rankings(rankings: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    rows_updated = 0
    today = date.today().isoformat()
    with engine.begin() as conn:
        existing = pd.read_sql("SELECT name FROM teams", conn)
        existing_names = set(existing["name"].tolist())
        for _, row in rankings.iterrows():
            team = row["team"]
            if team not in existing_names:
                continue
            result = conn.execute(
                text(
                    """
                    UPDATE teams
                    SET elo_rating = :elo_rating,
                        fifa_rank = :rank,
                        updated_at = :updated_at
                    WHERE name = :team
                    """
                ),
                {
                    "elo_rating": float(row["elo_rating"]),
                    "rank": int(row["rank"]),
                    "updated_at": today,
                    "team": team,
                },
            )
            rows_updated += int(result.rowcount or 0)
    return rows_updated


def _parse_published_date(html: str) -> date:
    match = re.search(r"World football Elo ratings<em>as on ([^<]+)</em>", html)
    if not match:
        return date.today()
    text_value = match.group(1).strip()
    for suffix in ("st", "nd", "rd", "th"):
        text_value = text_value.replace(suffix + ",", ",")
    try:
        return datetime.strptime(text_value, "%B %d, %Y").date()
    except ValueError:
        return date.today()


def _clean_html_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("&amp;", "&")).strip()

