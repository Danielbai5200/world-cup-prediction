from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.database import initialize_database


def update_team_data() -> None:
    initialize_database()


def update_player_data() -> None:
    initialize_database()


def update_odds_data() -> None:
    initialize_database()


def run_daily_update() -> dict[str, str]:
    update_team_data()
    update_player_data()
    update_odds_data()
    return {"status": "ok", "updated_at": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    result = run_daily_update()
    print(result)


if __name__ == "__main__":
    main()

