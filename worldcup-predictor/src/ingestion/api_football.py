from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy.engine import Engine

from src.ingestion.database import get_engine, initialize_database
from src.ingestion.source_mapping import load_team_source_mapping
from src.ingestion.transfermarkt import apply_player_snapshot
from src.utils.config import DATABASE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
USER_AGENT = "WorldCupPredictor2026/1.0 (+api-football updater)"


@dataclass(frozen=True)
class ApiFootballUpdateResult:
    source: str
    raw_dir: Path
    processed_path: Path
    teams_requested: int
    teams_updated: int
    rows_downloaded: int
    rows_updated: int
    updated_at: str
    metadata: dict[str, object]


def api_football_configured() -> bool:
    return bool(os.getenv("API_FOOTBALL_KEY", "").strip())


def fetch_api_football_json(endpoint: str, params: dict[str, object] | None = None, api_key: str | None = None) -> dict:
    key = api_key or os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise ValueError("API_FOOTBALL_KEY is not configured.")
    query = f"?{urlencode(params or {})}" if params else ""
    request = Request(
        f"{API_FOOTBALL_BASE_URL}/{endpoint.lstrip('/')}{query}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "x-apisports-key": key,
        },
    )
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    errors = data.get("errors") if isinstance(data, dict) else None
    if errors:
        raise ValueError(f"API-Football returned errors: {errors}")
    return data


def discover_api_football_team_id(team_name: str, api_key: str | None = None) -> int | None:
    for params in ({"name": team_name}, {"search": team_name}):
        data = fetch_api_football_json("teams", params, api_key)
        for item in data.get("response", []):
            team = item.get("team", {})
            if not team.get("national"):
                continue
            name = str(team.get("name") or "")
            if _normalize_name(name) == _normalize_name(team_name):
                return int(team["id"])
        for item in data.get("response", []):
            team = item.get("team", {})
            if team.get("national") and team.get("id"):
                return int(team["id"])
    return None


def fetch_squad(team_id: int, api_key: str | None = None) -> dict:
    return fetch_api_football_json("players/squads", {"team": team_id}, api_key)


def fetch_injuries(team_id: int, season: int, api_key: str | None = None) -> dict:
    return fetch_api_football_json("injuries", {"team": team_id, "season": season}, api_key)


def parse_squad_response(data: dict, team_name: str, injuries: dict | None = None) -> pd.DataFrame:
    injury_status_by_name = _injury_status_by_name(injuries or {})
    rows: list[dict[str, object]] = []
    for team_block in data.get("response", []):
        for player in team_block.get("players", []):
            name = str(player.get("name") or "").strip()
            if not name:
                continue
            status = injury_status_by_name.get(_normalize_name(name), "fit")
            rows.append(
                {
                    "name": name,
                    "team": team_name,
                    "position": _normalize_position(str(player.get("position") or "")),
                    "age": _safe_int(player.get("age")),
                    "market_value": None,
                    "form_score": 70.0,
                    "fitness_score": _fitness_score(status),
                    "injury_status": status,
                    "updated_at": date.today(),
                    "source": "api-football",
                    "api_football_player_id": player.get("id"),
                }
            )
    return pd.DataFrame(rows)


def update_api_football_player_data(
    engine: Engine | None = None,
    max_teams: int | None = None,
) -> ApiFootballUpdateResult:
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        return _skipped_result("API_FOOTBALL_KEY is not configured.")

    engine = engine or get_engine()
    if not DATABASE_PATH.exists():
        initialize_database()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc)
    raw_dir = RAW_DATA_DIR / f"api_football_{fetched_at:%Y%m%d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_team_source_mapping()
    if max_teams is not None:
        mapping = mapping.head(max_teams)

    season = int(os.getenv("API_FOOTBALL_SEASON", "2026"))
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    discovered_ids: dict[str, int] = {}

    for _, row in mapping.iterrows():
        team_name = str(row["team"])
        team_id = _mapping_team_id(row)
        try:
            if team_id is None:
                team_id = discover_api_football_team_id(team_name, api_key)
            if team_id is None:
                failures[team_name] = "team id not found"
                continue
            discovered_ids[team_name] = team_id
            squad_data = fetch_squad(team_id, api_key)
            injuries_data = fetch_injuries(team_id, season, api_key)
            (raw_dir / f"{_safe_filename(team_name)}_squad.json").write_text(
                json.dumps(squad_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (raw_dir / f"{_safe_filename(team_name)}_injuries.json").write_text(
                json.dumps(injuries_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            players = parse_squad_response(squad_data, team_name, injuries_data)
            if players.empty:
                failures[team_name] = "parsed 0 players"
                continue
            frames.append(players)
        except Exception as exc:  # pragma: no cover - defensive network fallback
            failures[team_name] = f"{type(exc).__name__}: {exc}"

    players = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    processed_path = PROCESSED_DATA_DIR / "api_football_players_latest.csv"
    if not players.empty:
        players.to_csv(processed_path, index=False)
        rows_updated = apply_player_snapshot(players, engine)
    else:
        processed_path.write_text("", encoding="utf-8")
        rows_updated = 0

    return ApiFootballUpdateResult(
        source="API-Football",
        raw_dir=raw_dir,
        processed_path=processed_path,
        teams_requested=len(mapping),
        teams_updated=len(frames),
        rows_downloaded=len(players),
        rows_updated=rows_updated,
        updated_at=fetched_at.isoformat(),
        metadata={"season": season, "failures": failures, "discovered_team_ids": discovered_ids},
    )


def _skipped_result(reason: str) -> ApiFootballUpdateResult:
    path = PROCESSED_DATA_DIR / "api_football_players_latest.csv"
    return ApiFootballUpdateResult(
        source="API-Football",
        raw_dir=RAW_DATA_DIR,
        processed_path=path,
        teams_requested=0,
        teams_updated=0,
        rows_downloaded=0,
        rows_updated=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
        metadata={"status": "skipped", "reason": reason},
    )


def _mapping_team_id(row: pd.Series) -> int | None:
    value = row.get("api_football_team_id", "")
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(float(value))


def _injury_status_by_name(data: dict) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in data.get("response", []):
        player = item.get("player", {})
        name = str(player.get("name") or "").strip()
        if not name:
            continue
        status_type = str(item.get("type") or item.get("reason") or "").lower()
        if "question" in status_type:
            status = "doubtful"
        elif "suspend" in status_type:
            status = "out"
        else:
            status = "injured"
        statuses[_normalize_name(name)] = status
    return statuses


def _fitness_score(status: str) -> float:
    return {"fit": 90.0, "doubtful": 65.0, "injured": 45.0, "out": 35.0}.get(status, 70.0)


def _normalize_position(value: str) -> str:
    lower = value.lower()
    if "goal" in lower:
        return "GK"
    if "def" in lower or "back" in lower:
        return "DF"
    if "mid" in lower:
        return "MF"
    if "att" in lower or "forward" in lower or "wing" in lower:
        return "FW"
    return value[:2].upper() if value else "MF"


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    return value.casefold().replace("&", "and").replace("ı", "i").strip()


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")
