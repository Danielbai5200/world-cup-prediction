from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from src.utils.config import CONFIG_DATA_DIR


TEAM_SOURCE_MAPPING_PATH = CONFIG_DATA_DIR / "team_source_mapping.csv"
ONEFOOTBALL_SEARCH_API_URL = "https://search-api.onefootball.com/v2/en/search"
REQUIRED_COLUMNS = {
    "team",
    "fifa_name",
    "fbref_squad_id",
    "fbref_slug",
    "fbref_stats_url",
    "fbref_history_url",
    "onefootball_url",
    "onefootball_status",
}


@dataclass(frozen=True)
class MappingValidationResult:
    teams: int
    fbref_urls: int
    onefootball_urls: int
    missing_fbref: list[str]
    missing_onefootball: list[str]


def load_team_source_mapping(path=TEAM_SOURCE_MAPPING_PATH) -> pd.DataFrame:
    mapping = pd.read_csv(path).fillna("")
    missing_columns = REQUIRED_COLUMNS - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"Team source mapping is missing columns: {sorted(missing_columns)}")
    if mapping["team"].duplicated().any():
        duplicated = mapping.loc[mapping["team"].duplicated(), "team"].tolist()
        raise ValueError(f"Team source mapping contains duplicate teams: {duplicated}")
    return mapping


def validate_team_source_mapping(mapping: pd.DataFrame | None = None) -> MappingValidationResult:
    mapping = mapping if mapping is not None else load_team_source_mapping()
    missing_fbref = mapping.loc[mapping["fbref_stats_url"].eq(""), "team"].tolist()
    missing_onefootball = mapping.loc[mapping["onefootball_url"].eq(""), "team"].tolist()
    return MappingValidationResult(
        teams=len(mapping),
        fbref_urls=int(mapping["fbref_stats_url"].ne("").sum()),
        onefootball_urls=int(mapping["onefootball_url"].ne("").sum()),
        missing_fbref=missing_fbref,
        missing_onefootball=missing_onefootball,
    )


def check_url_available(url: str, timeout: int = 10) -> bool:
    if not url:
        return False
    request = Request(url, headers={"User-Agent": "WorldCupPredictor2026/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return 200 <= response.status < 400


def discover_onefootball_team(team: str, timeout: int = 20) -> dict[str, str] | None:
    url = f"{ONEFOOTBALL_SEARCH_API_URL}?q={quote(team)}"
    request = Request(url, headers={"User-Agent": "WorldCupPredictor2026/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    for item in data.get("teams", []):
        if item.get("is_national") and item.get("name") == team and item.get("url"):
            relative_url = item["url"]
            slug_id = relative_url.rstrip("/").split("/")[-1]
            slug, _, team_id = slug_id.rpartition("-")
            return {
                "team": team,
                "onefootball_slug": slug,
                "onefootball_team_id": str(item["id"] or team_id),
                "onefootball_url": f"https://onefootball.com{relative_url}",
                "onefootball_status": "verified_search_api",
            }
    return None
