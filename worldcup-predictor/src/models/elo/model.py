from __future__ import annotations

import math

import pandas as pd


class EloModel:
    def __init__(self, teams: pd.DataFrame, draw_base: float = 0.24):
        self.ratings = teams.set_index("name")["elo_rating"].astype(float).to_dict()
        self.draw_base = draw_base

    def predict_outcome(self, home_team: str, away_team: str) -> dict[str, float]:
        home = self.ratings[home_team]
        away = self.ratings[away_team]
        expected_home = 1 / (1 + math.pow(10, -(home + 45 - away) / 400))
        rating_gap = abs(home - away)
        draw = max(0.16, self.draw_base - min(rating_gap, 250) / 2500)
        non_draw = 1 - draw
        home_win = expected_home * non_draw
        away_win = (1 - expected_home) * non_draw
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

