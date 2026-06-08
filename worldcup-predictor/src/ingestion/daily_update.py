from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.database import initialize_database
from src.ingestion.real_data_update import update_elo_data


def update_team_data() -> dict[str, object]:
    initialize_database()
    result = update_elo_data()
    return {
        "source": result.source,
        "rows_downloaded": result.rows_downloaded,
        "rows_updated": result.rows_updated,
        "raw_path": str(result.raw_path),
        "processed_path": str(result.processed_path),
    }


def update_player_data() -> dict[str, object]:
    return {"status": "skipped", "reason": "真实球员数据源尚未配置，保留当前数据库数据。"}


def update_odds_data() -> dict[str, object]:
    return {"status": "skipped", "reason": "真实赔率数据源尚未配置，保留当前数据库数据。"}


def run_daily_update() -> dict[str, object]:
    team_update = update_team_data()
    player_update = update_player_data()
    odds_update = update_odds_data()
    return {
        "status": "ok",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "team_update": team_update,
        "player_update": player_update,
        "odds_update": odds_update,
    }


def main() -> None:
    result = run_daily_update()
    print(result)


if __name__ == "__main__":
    main()
