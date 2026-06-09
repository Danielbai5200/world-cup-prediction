from __future__ import annotations

from datetime import datetime, timezone
import os

from src.ingestion.api_football import api_football_configured, update_api_football_player_data
from src.ingestion.database import initialize_database
from src.ingestion.odds_sources import update_match_odds_data, update_prediction_market_odds
from src.ingestion.real_data_update import update_elo_data
from src.ingestion.source_mapping import load_team_source_mapping, validate_team_source_mapping
from src.ingestion.transfermarkt import update_transfermarkt_player_data
from src.utils.env import load_local_env


def update_team_data() -> dict[str, object]:
    initialize_database()
    result = update_elo_data()
    return {
        "source": result.source,
        "rows_downloaded": result.rows_downloaded,
        "rows_updated": result.rows_updated,
        "raw_path": str(result.raw_path),
        "processed_path": str(result.processed_path),
        "metadata": result.metadata,
    }


def update_player_data() -> dict[str, object]:
    api_max_teams_env = os.getenv("API_FOOTBALL_MAX_TEAMS", "").strip()
    api_max_teams = int(api_max_teams_env) if api_max_teams_env else None
    if api_football_configured():
        try:
            api_result = update_api_football_player_data(max_teams=api_max_teams)
        except Exception as exc:  # pragma: no cover - defensive network fallback
            api_result = None
            api_failure = f"{type(exc).__name__}: {exc}"
        else:
            api_failure = ""
        if api_result is not None and api_result.rows_updated:
            return _player_result_dict("ok", api_result)
        if api_result is not None:
            api_failure = str(api_result.metadata.get("failures") or api_result.metadata)
    else:
        api_failure = "API_FOOTBALL_KEY is not configured."

    max_teams_env = os.getenv("TRANSFERMARKT_MAX_TEAMS", "").strip()
    max_teams = int(max_teams_env) if max_teams_env else None
    delay = float(os.getenv("TRANSFERMARKT_DELAY_SECONDS", "3"))
    try:
        result = update_transfermarkt_player_data(max_teams=max_teams, request_delay_seconds=delay)
    except Exception as exc:  # pragma: no cover - defensive network fallback
        return {
            "status": "failed",
            "source": "Transfermarkt",
            "primary_source": "API-Football",
            "primary_status": "failed",
            "primary_reason": api_failure,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    fallback = {
        "status": "ok" if result.rows_updated else "skipped",
        "source": result.source,
        "teams_requested": result.teams_requested,
        "teams_updated": result.teams_updated,
        "rows_downloaded": result.rows_downloaded,
        "rows_updated": result.rows_updated,
        "raw_dir": str(result.raw_dir),
        "processed_path": str(result.processed_path),
        "metadata": result.metadata,
    }
    fallback["primary_source"] = "API-Football"
    fallback["primary_status"] = "failed" if api_football_configured() else "skipped"
    fallback["primary_reason"] = api_failure
    return fallback


def update_odds_data() -> dict[str, object]:
    match_result = update_match_odds_data()
    market_result = update_prediction_market_odds()
    status = "ok" if match_result.rows_updated or market_result.rows_updated else "skipped"
    return {
        "status": status,
        "match_odds": {
            "source": match_result.source,
            "rows_downloaded": match_result.rows_downloaded,
            "rows_updated": match_result.rows_updated,
            "processed_path": str(match_result.processed_path),
            "metadata": match_result.metadata,
        },
        "winner_market": {
            "source": market_result.source,
            "rows_downloaded": market_result.rows_downloaded,
            "rows_updated": market_result.rows_updated,
            "processed_path": str(market_result.processed_path),
            "metadata": market_result.metadata,
        },
    }


def _player_result_dict(status: str, result) -> dict[str, object]:
    return {
        "status": status,
        "source": result.source,
        "teams_requested": result.teams_requested,
        "teams_updated": result.teams_updated,
        "rows_downloaded": result.rows_downloaded,
        "rows_updated": result.rows_updated,
        "raw_dir": str(result.raw_dir),
        "processed_path": str(result.processed_path),
        "metadata": result.metadata,
    }


def run_daily_update() -> dict[str, object]:
    load_local_env()
    team_update = update_team_data()
    player_update = update_player_data()
    odds_update = update_odds_data()
    mapping_status = validate_team_source_mapping(load_team_source_mapping())
    return {
        "status": "ok",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "team_update": team_update,
        "player_update": player_update,
        "odds_update": odds_update,
        "source_mapping": {
            "teams": mapping_status.teams,
            "fbref_urls": mapping_status.fbref_urls,
            "onefootball_urls": mapping_status.onefootball_urls,
            "transfermarkt_urls": mapping_status.transfermarkt_urls,
            "missing_fbref": mapping_status.missing_fbref,
            "missing_onefootball": mapping_status.missing_onefootball,
            "missing_transfermarkt": mapping_status.missing_transfermarkt,
        },
    }


def main() -> None:
    result = run_daily_update()
    print(result)


if __name__ == "__main__":
    main()
