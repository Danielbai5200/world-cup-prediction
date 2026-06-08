from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.utils.config import SAMPLE_DATA_DIR


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


class FBrefSource(ExternalProviderStub):
    provider_name = "FBref"


class TransfermarktSource(ExternalProviderStub):
    provider_name = "Transfermarkt"


class StatsBombSource(ExternalProviderStub):
    provider_name = "StatsBomb"


class OddsApiSource(ExternalProviderStub):
    provider_name = "Odds API"

