from __future__ import annotations

import numpy as np

from src.models.predictor import MatchPredictor


def test_match_predictor_returns_probabilities_and_scores() -> None:
    predictor = MatchPredictor()
    prediction = predictor.predict_match("Argentina", "France")
    probabilities = prediction["probabilities"]
    assert set(probabilities) == {"home_win", "draw", "away_win"}
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert len(prediction["top_scores"]) == 10
    assert prediction["score_matrix"].shape == (7, 7)


def test_poisson_score_matrix_is_normalized() -> None:
    predictor = MatchPredictor()
    matrix = predictor.poisson_model.score_matrix("Brazil", "Spain")
    assert matrix.shape == (7, 7)
    assert np.isclose(matrix.to_numpy().sum(), 1.0)


def test_same_team_prediction_is_rejected() -> None:
    predictor = MatchPredictor()
    try:
        predictor.predict_match("Japan", "Japan")
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("Expected ValueError for same-team match.")

