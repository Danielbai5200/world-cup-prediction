from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.predictor import MatchPredictor


@dataclass(frozen=True)
class SimulatedMatch:
    home_team: str
    away_team: str
    winner: str
    loser: str
    is_draw: bool


class WorldCupSimulator:
    def __init__(self, predictor: MatchPredictor | None = None, random_seed: int | None = 2026):
        self.predictor = predictor or MatchPredictor()
        self.rng = np.random.default_rng(random_seed)
        self._probability_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._xg_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def simulate(self, n_simulations: int = 100_000) -> pd.DataFrame:
        teams = self._expanded_teams()
        counters = {
            team: {
                "group_qualified": 0,
                "round_32": 0,
                "round_16": 0,
                "quarterfinal": 0,
                "semifinal": 0,
                "final": 0,
                "champion": 0,
            }
            for team in teams
        }
        for _ in range(n_simulations):
            groups = self._make_groups(teams)
            round_32 = self._simulate_group_stage(groups)
            for team in round_32:
                counters[team]["group_qualified"] += 1
                counters[team]["round_32"] += 1
            round_16 = self._simulate_knockout_round(round_32)
            self._increment(counters, round_16, "round_16")
            quarterfinal = self._simulate_knockout_round(round_16)
            self._increment(counters, quarterfinal, "quarterfinal")
            semifinal = self._simulate_knockout_round(quarterfinal)
            self._increment(counters, semifinal, "semifinal")
            finalists = self._simulate_knockout_round(semifinal)
            self._increment(counters, finalists, "final")
            champion = self._simulate_knockout_round(finalists)[0]
            counters[champion]["champion"] += 1
        rows = []
        for team, values in counters.items():
            rows.append({"team": team, **{key: value / n_simulations for key, value in values.items()}})
        result = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
        return result

    def _expanded_teams(self) -> list[str]:
        base = self.predictor.team_names
        if len(base) >= 48:
            return base[:48]
        fillers = [f"Qualifier {idx:02d}" for idx in range(1, 49 - len(base))]
        return base + fillers

    def _make_groups(self, teams: list[str]) -> list[list[str]]:
        shuffled = list(self.rng.permutation(teams))
        return [shuffled[idx : idx + 4] for idx in range(0, 48, 4)]

    def _simulate_group_stage(self, groups: list[list[str]]) -> list[str]:
        qualified: list[str] = []
        third_place: list[tuple[str, int, int, int]] = []
        for group in groups:
            table = {team: {"points": 0, "gf": 0, "ga": 0} for team in group}
            for idx, home in enumerate(group):
                for away in group[idx + 1 :]:
                    match = self._simulate_match(home, away, allow_draw=True)
                    home_goals, away_goals = self._sample_score(home, away)
                    table[home]["gf"] += home_goals
                    table[home]["ga"] += away_goals
                    table[away]["gf"] += away_goals
                    table[away]["ga"] += home_goals
                    if match.is_draw:
                        table[home]["points"] += 1
                        table[away]["points"] += 1
                    else:
                        table[match.winner]["points"] += 3
            ranked = sorted(
                group,
                key=lambda team: (
                    table[team]["points"],
                    table[team]["gf"] - table[team]["ga"],
                    table[team]["gf"],
                    self._team_strength(team),
                ),
                reverse=True,
            )
            qualified.extend(ranked[:2])
            third = ranked[2]
            third_place.append(
                (
                    third,
                    table[third]["points"],
                    table[third]["gf"] - table[third]["ga"],
                    table[third]["gf"],
                )
            )
        best_thirds = sorted(third_place, key=lambda row: (row[1], row[2], row[3], self._team_strength(row[0])), reverse=True)[:8]
        qualified.extend(team for team, *_ in best_thirds)
        return list(self.rng.permutation(qualified))

    def _simulate_knockout_round(self, teams: list[str]) -> list[str]:
        winners = []
        bracket = list(teams)
        for idx in range(0, len(bracket), 2):
            winners.append(self._simulate_match(bracket[idx], bracket[idx + 1], allow_draw=False).winner)
        return winners

    def _simulate_match(self, home_team: str, away_team: str, allow_draw: bool) -> SimulatedMatch:
        probabilities = self._match_probabilities(home_team, away_team)
        if allow_draw:
            outcomes = ["home", "draw", "away"]
            probs = [probabilities["home_win"], probabilities["draw"], probabilities["away_win"]]
            sampled = self.rng.choice(outcomes, p=np.array(probs) / sum(probs))
            if sampled == "draw":
                tiebreak = self._choose_by_strength(home_team, away_team)
                return SimulatedMatch(home_team, away_team, tiebreak, away_team if tiebreak == home_team else home_team, True)
            winner = home_team if sampled == "home" else away_team
        else:
            probs = np.array([probabilities["home_win"] + 0.5 * probabilities["draw"], probabilities["away_win"] + 0.5 * probabilities["draw"]])
            winner = self.rng.choice([home_team, away_team], p=probs / probs.sum())
        loser = away_team if winner == home_team else home_team
        return SimulatedMatch(home_team, away_team, str(winner), loser, False)

    def _match_probabilities(self, home_team: str, away_team: str) -> dict[str, float]:
        cache_key = (home_team, away_team)
        if cache_key in self._probability_cache:
            return self._probability_cache[cache_key]
        if home_team in self.predictor.team_names and away_team in self.predictor.team_names:
            probabilities = self.predictor.predict_match(home_team, away_team)["probabilities"]  # type: ignore[assignment]
            self._probability_cache[cache_key] = probabilities
            return probabilities
        home_strength = self._team_strength(home_team)
        away_strength = self._team_strength(away_team)
        home_win = 1 / (1 + np.exp(-(home_strength - away_strength) / 10))
        draw = 0.22
        probabilities = {"home_win": float(home_win * (1 - draw)), "draw": draw, "away_win": float((1 - home_win) * (1 - draw))}
        self._probability_cache[cache_key] = probabilities
        return probabilities

    def _sample_score(self, home_team: str, away_team: str) -> tuple[int, int]:
        cache_key = (home_team, away_team)
        if home_team in self.predictor.team_names and away_team in self.predictor.team_names:
            if cache_key not in self._xg_cache:
                self._xg_cache[cache_key] = self.predictor.poisson_model.expected_goals(home_team, away_team)
            home_xg, away_xg = self._xg_cache[cache_key]
        else:
            home_xg = max(0.4, 1.25 + (self._team_strength(home_team) - self._team_strength(away_team)) / 55)
            away_xg = max(0.3, 1.10 + (self._team_strength(away_team) - self._team_strength(home_team)) / 60)
        return int(min(self.rng.poisson(home_xg), 8)), int(min(self.rng.poisson(away_xg), 8))

    def _choose_by_strength(self, team_a: str, team_b: str) -> str:
        prob_a = 1 / (1 + np.exp(-(self._team_strength(team_a) - self._team_strength(team_b)) / 10))
        return str(self.rng.choice([team_a, team_b], p=[prob_a, 1 - prob_a]))

    def _team_strength(self, team: str) -> float:
        features = self.predictor.team_features
        row = features.loc[features["name"] == team]
        if not row.empty:
            return float(row.iloc[0]["team_rating"])
        seed = int(team.split()[-1]) if team.startswith("Qualifier") else 24
        return max(15.0, 52.0 - seed * 0.7)

    @staticmethod
    def _increment(counters: dict[str, dict[str, int]], teams: list[str], stage: str) -> None:
        for team in teams:
            counters[team][stage] += 1
