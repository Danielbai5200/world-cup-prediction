from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.ingestion.database import get_engine
from src.ingestion.odds_sources import fetch_the_odds_api_match_odds
from src.ingestion.transfermarkt import parse_transfermarkt_squad, fetch_transfermarkt_html
from src.utils.config import DATABASE_PATH, SAMPLE_DATA_DIR


class FootballDataSource(ABC):
    @abstractmethod
    def teams(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def players(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def matches(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def odds(self) -> pd.DataFrame:
        raise NotImplementedError


class CsvSampleDataSource(FootballDataSource):
    def __init__(self, data_dir: Path = SAMPLE_DATA_DIR):
        self.data_dir = data_dir

    def teams(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "teams.csv", parse_dates=["updated_at"])

    def players(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "players.csv", parse_dates=["updated_at"])

    def matches(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "matches.csv", parse_dates=["date"])

    def odds(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "odds.csv", parse_dates=["timestamp"])


class SQLiteDataSource(FootballDataSource):
    def __init__(self, database_path: Path = DATABASE_PATH):
        self.database_path = database_path

    @property
    def available(self) -> bool:
        return self.database_path.exists()

    def teams(self) -> pd.DataFrame:
        return self._read_table("teams")

    def players(self) -> pd.DataFrame:
        return self._read_table("players")

    def matches(self) -> pd.DataFrame:
        return self._read_table("matches")

    def odds(self) -> pd.DataFrame:
        return self._read_table("odds")

    def _read_table(self, table: str) -> pd.DataFrame:
        if not self.available:
            raise FileNotFoundError(f"Database does not exist: {self.database_path}")
        engine = get_engine(f"sqlite:///{self.database_path}")
        return pd.read_sql_table(table, engine)


class ExternalProviderStub(FootballDataSource):
    provider_name = "external"

    def _not_configured(self) -> pd.DataFrame:
        raise NotImplementedError(f"{self.provider_name} integration is reserved for a future release.")

    def teams(self) -> pd.DataFrame:
        return self._not_configured()

    def players(self) -> pd.DataFrame:
        return self._not_configured()

    def matches(self) -> pd.DataFrame:
        return self._not_configured()

    def odds(self) -> pd.DataFrame:
        return self._not_configured()


class WorldFootballEloSource(ExternalProviderStub):
    provider_name = "World Football Elo Ratings"


class FIFAOfficialRankingsSource(ExternalProviderStub):
    provider_name = "FIFA official rankings"


class FBrefSource(ExternalProviderStub):
    provider_name = "FBref"


class OneFootballSource(ExternalProviderStub):
    provider_name = "OneFootball"


class TransfermarktSource(FootballDataSource):
    provider_name = "Transfermarkt"

    def __init__(self, team_urls: dict[str, str] | None = None):
        self.team_urls = team_urls or {}

    def teams(self) -> pd.DataFrame:
        return pd.DataFrame()

    def players(self) -> pd.DataFrame:
        frames = []
        for team, url in self.team_urls.items():
            frames.append(parse_transfermarkt_squad(fetch_transfermarkt_html(url), team))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def matches(self) -> pd.DataFrame:
        return pd.DataFrame()

    def odds(self) -> pd.DataFrame:
        return pd.DataFrame()


class StatsBombSource(ExternalProviderStub):
    provider_name = "StatsBomb"


class OddsApiSource(FootballDataSource):
    provider_name = "Odds API"

    def __init__(self, api_key: str, sport_key: str = "soccer_fifa_world_cup"):
        self.api_key = api_key
        self.sport_key = sport_key

    def teams(self) -> pd.DataFrame:
        return pd.DataFrame()

    def players(self) -> pd.DataFrame:
        return pd.DataFrame()

    def matches(self) -> pd.DataFrame:
        return pd.DataFrame()

    def odds(self) -> pd.DataFrame:
        return fetch_the_odds_api_match_odds(self.api_key, self.sport_key)
