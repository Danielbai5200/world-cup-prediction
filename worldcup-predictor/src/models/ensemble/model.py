from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloModel
from src.models.poisson import PoissonModel


class EnsembleModel:
    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        dixon_coles_model: DixonColesModel,
        odds: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.elo_model = elo_model
        self.poisson_model = poisson_model
        self.dixon_coles_model = dixon_coles_model
        self.odds = odds
        self.weights = weights or {"elo": 0.30, "poisson": 0.25, "dixon_coles": 0.30, "odds": 0.15}

    def predict_outcome(self, home_team: str, away_team: str, odds_row: pd.Series | None = None) -> dict[str, float]:
        components = {
            "elo": self.elo_model.predict_outcome(home_team, away_team),
            "poisson": self.poisson_model.predict_outcome(home_team, away_team),
            "dixon_coles": self.dixon_coles_model.predict_outcome(home_team, away_team),
        }
        if odds_row is not None:
            components["odds"] = odds_to_probabilities(odds_row)
        else:
            components["odds"] = components["poisson"]
        raw = {
            outcome: sum(self.weights[name] * components[name][outcome] for name in self.weights)
            for outcome in ("home_win", "draw", "away_win")
        }
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}


def odds_to_probabilities(odds_row: pd.Series) -> dict[str, float]:
    implied = np.array(
        [
            1 / float(odds_row["home_win_odds"]),
            1 / float(odds_row["draw_odds"]),
            1 / float(odds_row["away_win_odds"]),
        ]
    )
    implied = implied / implied.sum()
    return {"home_win": float(implied[0]), "draw": float(implied[1]), "away_win": float(implied[2])}

