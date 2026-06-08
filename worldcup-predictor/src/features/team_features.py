from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def build_team_features(
    teams: pd.DataFrame,
    players: pd.DataFrame,
    matches: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = teams.copy()
    recent = _recent_match_features(matches)
    player = _player_features(players)
    injury = _injury_features(injuries if injuries is not None else players)
    features = features.merge(recent, left_on="name", right_index=True, how="left")
    features = features.merge(player, left_on="name", right_index=True, how="left")
    features = features.merge(injury, left_on="name", right_index=True, how="left")
    features = features.fillna(
        {
            "recent_points_per_match": 1.3,
            "goals_for_per_match": 1.4,
            "goals_against_per_match": 1.1,
            "xg_for_per_match": 1.4,
            "xg_against_per_match": 1.1,
            "player_form": 75.0,
            "player_fitness": 82.0,
            "injury_penalty": 0.0,
        }
    )
    features["team_rating"] = (
        45 * _minmax(features["elo_rating"])
        + 20 * _minmax(features["market_value"])
        + 20 * _minmax(features["recent_points_per_match"])
        + 10 * (1 - _minmax(features["fifa_rank"]))
        + 5 * (1 - _minmax(features["injury_penalty"]))
    )
    features["attack_feature"] = (
        35 * _minmax(features["goals_for_per_match"])
        + 35 * _minmax(features["xg_for_per_match"])
        + 15 * _minmax(features["attack_rating"])
        + 15 * _minmax(features["player_form"])
    )
    features["defense_feature"] = (
        35 * (1 - _minmax(features["goals_against_per_match"]))
        + 35 * (1 - _minmax(features["xg_against_per_match"]))
        + 20 * _minmax(features["defense_rating"])
        + 10 * _minmax(features["player_fitness"])
    )
    features["form_rating"] = (
        45 * _minmax(features["recent_points_per_match"])
        + 25 * _minmax(features["player_form"])
        + 20 * _minmax(features["player_fitness"])
        + 10 * (1 - _minmax(features["injury_penalty"]))
    )
    return features


def _recent_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    sorted_matches = matches.sort_values("date").tail(60)
    for _, row in sorted_matches.iterrows():
        home_points = 3 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 0
        away_points = 3 if row.away_goals > row.home_goals else 1 if row.home_goals == row.away_goals else 0
        records.append(
            {
                "team": row.home_team,
                "points": home_points,
                "gf": row.home_goals,
                "ga": row.away_goals,
                "xgf": row.xg_home,
                "xga": row.xg_away,
            }
        )
        records.append(
            {
                "team": row.away_team,
                "points": away_points,
                "gf": row.away_goals,
                "ga": row.home_goals,
                "xgf": row.xg_away,
                "xga": row.xg_home,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("team").agg(
        recent_points_per_match=("points", "mean"),
        goals_for_per_match=("gf", "mean"),
        goals_against_per_match=("ga", "mean"),
        xg_for_per_match=("xgf", "mean"),
        xg_against_per_match=("xga", "mean"),
    )
    return grouped


def _player_features(players: pd.DataFrame) -> pd.DataFrame:
    return players.groupby("team").agg(player_form=("form_score", "mean"), player_fitness=("fitness_score", "mean"))


def _injury_features(injury_source: pd.DataFrame) -> pd.DataFrame:
    status_penalty = {"fit": 0, "minor": 2, "doubtful": 5, "injured": 10, "out": 12}
    if "severity" in injury_source.columns:
        injury_source = injury_source.rename(columns={"player_name": "name"}).copy()
        injury_source["injury_penalty"] = injury_source["severity"].fillna(0).astype(float)
    else:
        injury_source = injury_source.copy()
        injury_source["injury_penalty"] = injury_source["injury_status"].map(status_penalty).fillna(0)
    return injury_source.groupby("team").agg(injury_penalty=("injury_penalty", "sum"))

