from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion.database import get_engine, initialize_database
from src.ingestion.source_mapping import load_team_source_mapping
from src.utils.config import DATABASE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TRANSFERMARKT_SOURCE = "https://www.transfermarkt.com"


class ProviderBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlayerUpdateResult:
    source: str
    raw_dir: Path
    processed_path: Path
    teams_requested: int
    teams_updated: int
    rows_downloaded: int
    rows_updated: int
    updated_at: str
    metadata: dict[str, object]


def fetch_transfermarkt_html(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        html_text = response.read().decode("utf-8", errors="replace")
    if _looks_like_waf_challenge(html_text):
        raise ProviderBlockedError("Transfermarkt returned a human-verification challenge.")
    return html_text


def parse_transfermarkt_squad(html_text: str, team: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_html in _iter_player_rows(html_text):
        name = _extract_player_name(row_html)
        if not name:
            continue
        position = _normalize_position(_extract_position(row_html))
        age = _extract_age(row_html)
        market_value = _parse_market_value(_extract_market_value_text(row_html))
        injured = _looks_injured(row_html)
        rows.append(
            {
                "name": name,
                "team": team,
                "position": position,
                "age": age,
                "market_value": market_value,
                "form_score": 70.0,
                "fitness_score": 55.0 if injured else 90.0,
                "injury_status": "injured" if injured else "fit",
                "updated_at": date.today(),
                "source": "transfermarkt",
            }
        )
    return pd.DataFrame(rows)


def update_transfermarkt_player_data(
    engine: Engine | None = None,
    max_teams: int | None = None,
    request_delay_seconds: float = 3.0,
) -> PlayerUpdateResult:
    engine = engine or get_engine()
    if not DATABASE_PATH.exists():
        initialize_database()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    mapping = load_team_source_mapping()
    if max_teams is not None:
        mapping = mapping.head(max_teams)

    fetched_at = datetime.now(timezone.utc)
    raw_dir = RAW_DATA_DIR / f"transfermarkt_{fetched_at:%Y%m%d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for index, row in mapping.iterrows():
        team = str(row["team"])
        url = str(row["transfermarkt_url"])
        if not url:
            failures[team] = "missing transfermarkt_url"
            continue
        try:
            html_text = fetch_transfermarkt_html(url)
            (raw_dir / f"{_safe_filename(team)}.html").write_text(html_text, encoding="utf-8")
            squad = parse_transfermarkt_squad(html_text, team)
            if squad.empty:
                failures[team] = "parsed 0 players"
                continue
            frames.append(squad)
        except HTTPError as exc:  # pragma: no cover - network defensive path
            failures[team] = f"HTTPError: HTTP {exc.code}"
            if exc.code in {403, 405, 429, 503}:
                failures["_batch_stopped"] = "Transfermarkt appears to be blocking automated requests."
                break
        except ProviderBlockedError as exc:  # pragma: no cover - network defensive path
            failures[team] = f"{type(exc).__name__}: {exc}"
            failures["_batch_stopped"] = "Transfermarkt returned a human-verification challenge."
            break
        except Exception as exc:  # pragma: no cover - network defensive path
            failures[team] = f"{type(exc).__name__}: {exc}"
        if index != mapping.index[-1] and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    players = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    processed_path = PROCESSED_DATA_DIR / "transfermarkt_players_latest.csv"
    if not players.empty:
        players.to_csv(processed_path, index=False)
        rows_updated = apply_player_snapshot(players, engine)
    else:
        processed_path.write_text("", encoding="utf-8")
        rows_updated = 0

    return PlayerUpdateResult(
        source=TRANSFERMARKT_SOURCE,
        raw_dir=raw_dir,
        processed_path=processed_path,
        teams_requested=len(mapping),
        teams_updated=len(frames),
        rows_downloaded=len(players),
        rows_updated=rows_updated,
        updated_at=fetched_at.isoformat(),
        metadata={"failures": failures},
    )


def apply_player_snapshot(players: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    required = {
        "name",
        "team",
        "position",
        "age",
        "market_value",
        "form_score",
        "fitness_score",
        "injury_status",
        "updated_at",
    }
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Player snapshot missing columns: {sorted(missing)}")

    rows_updated = 0
    with engine.begin() as conn:
        for team, team_players in players.groupby("team"):
            conn.execute(text("DELETE FROM players WHERE team = :team"), {"team": team})
            write_df = team_players[list(required)].copy()
            write_df["updated_at"] = pd.to_datetime(write_df["updated_at"]).dt.date
            write_df.to_sql("players", conn, if_exists="append", index=False)
            rows_updated += len(write_df)
    return rows_updated


def _iter_player_rows(html_text: str) -> list[str]:
    starts = list(re.finditer(r"<tr[^>]*class=\"[^\"]*(?:odd|even)[^\"]*\"[^>]*>", html_text, re.S))
    rows = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else html_text.find("</tbody>", match.end())
        rows.append(html_text[match.start() : end if end != -1 else len(html_text)])
    return rows


def _extract_player_name(row_html: str) -> str:
    match = re.search(r"<table[^>]*class=\"inline-table\".*?<a[^>]*>(.*?)</a>", row_html, re.S)
    return _clean(match.group(1)) if match else ""


def _extract_position(row_html: str) -> str:
    match = re.search(r"<table[^>]*class=\"inline-table\".*?<tr>.*?</tr>\s*<tr>\s*<td[^>]*>(.*?)</td>", row_html, re.S)
    return _clean(match.group(1)) if match else ""


def _extract_age(row_html: str) -> int | None:
    text_value = _clean(row_html)
    match = re.search(r"\((\d{2})\)", text_value)
    return int(match.group(1)) if match else None


def _extract_market_value_text(row_html: str) -> str:
    matches = re.findall(r"<td[^>]*class=\"[^\"]*hauptlink[^\"]*\"[^>]*>(.*?)</td>", row_html, re.S)
    return _clean(matches[-1]) if matches else ""


def _parse_market_value(raw: str) -> float | None:
    if not raw or raw == "-":
        return None
    match = re.search(r"([\d.,]+)\s*([mk])", raw.lower())
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value / 1000 if match.group(2) == "k" else value


def _looks_injured(row_html: str) -> bool:
    lower = row_html.lower()
    markers = ("injur", "verletzt", "suspend", "ausrufezeichen", "red card", "yellow cards")
    return any(marker in lower for marker in markers)


def _looks_like_waf_challenge(html_text: str) -> bool:
    lower = html_text[:5000].lower()
    return (
        "human verification" in lower
        or "x-amzn-waf-action" in lower
        or "captcha" in lower
        or "challenge.js" in lower
    )


def _normalize_position(raw: str) -> str:
    value = raw.lower()
    if "goalkeeper" in value or value == "gk":
        return "GK"
    if "back" in value or "defender" in value:
        return "DF"
    if "midfield" in value:
        return "MF"
    if "forward" in value or "striker" in value or "winger" in value:
        return "FW"
    return "MF"


def _clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
