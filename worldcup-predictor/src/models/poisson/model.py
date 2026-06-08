from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


class PoissonModel:
    def __init__(self, team_features: pd.DataFrame, max_goals: int = 6):
        self.features = team_features.set_index("name")
        self.max_goals = max_goals

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home = self.features.loc[home_team]
        away = self.features.loc[away_team]
        home_xg = 1.35 + 0.018 * (home.attack_feature - away.defense_feature) + 0.12
        away_xg = 1.18 + 0.018 * (away.attack_feature - home.defense_feature)
        return float(np.clip(home_xg, 0.25, 3.8)), float(np.clip(away_xg, 0.2, 3.5))

    def score_matrix(self, home_team: str, away_team: str) -> pd.DataFrame:
        home_xg, away_xg = self.expected_goals(home_team, away_team)
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1), dtype=float)
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                matrix[home_goals, away_goals] = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
        matrix = matrix / matrix.sum()
        return pd.DataFrame(matrix, index=range(self.max_goals + 1), columns=range(self.max_goals + 1))

    def predict_outcome(self, home_team: str, away_team: str) -> dict[str, float]:
        matrix = self.score_matrix(home_team, away_team).to_numpy()
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def top_scores(self, home_team: str, away_team: str, n: int = 10) -> list[dict[str, float | str]]:
        matrix = self.score_matrix(home_team, away_team)
        rows = []
        for home_goals in matrix.index:
            for away_goals in matrix.columns:
                rows.append(
                    {
                        "score": f"{home_goals}-{away_goals}",
                        "home_goals": int(home_goals),
                        "away_goals": int(away_goals),
                        "probability": float(matrix.loc[home_goals, away_goals]),
                    }
                )
        return sorted(rows, key=lambda row: row["probability"], reverse=True)[:n]

