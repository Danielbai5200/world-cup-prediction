from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.database import initialize_database
from src.ingestion.real_data_update import update_elo_data
from src.ingestion.source_mapping import load_team_source_mapping, validate_team_source_mapping


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
    return {"status": "skipped", "reason": "真实球员数据源尚未配置，保留当前数据库数据。"}


def update_odds_data() -> dict[str, object]:
    return {"status": "skipped", "reason": "真实赔率数据源尚未配置，保留当前数据库数据。"}


def run_daily_update() -> dict[str, object]:
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
            "missing_onefootball": mapping_status.missing_onefootball,
        },
    }


def main() -> None:
    result = run_daily_update()
    print(result)


if __name__ == "__main__":
    main()
