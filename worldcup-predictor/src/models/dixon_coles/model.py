from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel


class DixonColesModel:
    def __init__(self, poisson_model: PoissonModel, rho: float = -0.08):
        self.poisson_model = poisson_model
        self.rho = rho

    def score_matrix(self, home_team: str, away_team: str) -> pd.DataFrame:
        matrix = self.poisson_model.score_matrix(home_team, away_team).copy()
        home_xg, away_xg = self.poisson_model.expected_goals(home_team, away_team)
        adjustments = {
            (0, 0): 1 - home_xg * away_xg * self.rho,
            (0, 1): 1 + home_xg * self.rho,
            (1, 0): 1 + away_xg * self.rho,
            (1, 1): 1 - self.rho,
        }
        for score, factor in adjustments.items():
            matrix.loc[score] *= max(0.01, factor)
        return matrix / matrix.to_numpy().sum()

    def predict_outcome(self, home_team: str, away_team: str) -> dict[str, float]:
        matrix = self.score_matrix(home_team, away_team).to_numpy()
        return {
            "home_win": float(np.tril(matrix, -1).sum()),
            "draw": float(np.trace(matrix)),
            "away_win": float(np.triu(matrix, 1).sum()),
        }
