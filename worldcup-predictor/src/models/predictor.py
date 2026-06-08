from __future__ import annotations

import pandas as pd

from src.features.team_features import build_team_features
from src.ingestion.data_sources import CsvSampleDataSource, SQLiteDataSource
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloModel
from src.models.ensemble import EnsembleModel
from src.models.poisson import PoissonModel
from src.utils.config import SAMPLE_DATA_DIR


class MatchPredictor:
    def __init__(
        self,
        teams: pd.DataFrame | None = None,
        players: pd.DataFrame | None = None,
        matches: pd.DataFrame | None = None,
        odds: pd.DataFrame | None = None,
        injuries: pd.DataFrame | None = None,
    ):
        sqlite_source = SQLiteDataSource()
        source = sqlite_source if sqlite_source.available else CsvSampleDataSource(SAMPLE_DATA_DIR)
        self.teams = teams if teams is not None else source.teams()
        self.players = players if players is not None else source.players()
        self.matches = matches if matches is not None else source.matches()
        self.odds = odds if odds is not None else source.odds()
        if injuries is None:
            injuries_path = SAMPLE_DATA_DIR / "injuries.csv"
            self.injuries = pd.read_csv(injuries_path, parse_dates=["updated_at", "expected_return"])
        else:
            self.injuries = injuries
        self.team_features = build_team_features(self.teams, self.players, self.matches, self.injuries)
        self.elo_model = EloModel(self.teams)
        self.poisson_model = PoissonModel(self.team_features)
        self.dixon_coles_model = DixonColesModel(self.poisson_model)
        self.ensemble_model = EnsembleModel(self.elo_model, self.poisson_model, self.dixon_coles_model, self.odds)

    @property
    def team_names(self) -> list[str]:
        return sorted(self.teams["name"].tolist())

    def predict_match(self, home_team: str, away_team: str) -> dict[str, object]:
        if home_team == away_team:
            raise ValueError("Home team and away team must be different.")
        home_xg, away_xg = self.poisson_model.expected_goals(home_team, away_team)
        score_matrix = self.dixon_coles_model.score_matrix(home_team, away_team)
        probabilities = self.ensemble_model.predict_outcome(home_team, away_team)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": probabilities,
            "expected_goals": {"home": home_xg, "away": away_xg},
            "top_scores": self.poisson_model.top_scores(home_team, away_team, n=10),
            "score_matrix": score_matrix,
        }
