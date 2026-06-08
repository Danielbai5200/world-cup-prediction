from __future__ import annotations

from src.models.predictor import MatchPredictor
from src.simulation import WorldCupSimulator


def test_world_cup_simulation_returns_stage_probabilities() -> None:
    simulator = WorldCupSimulator(MatchPredictor(), random_seed=7)
    result = simulator.simulate(n_simulations=100)
    assert len(result) == 48
    assert {"team", "group_qualified", "round_32", "round_16", "quarterfinal", "semifinal", "final", "champion"}.issubset(result.columns)
    assert abs(result["champion"].sum() - 1.0) < 1e-9
    assert result["group_qualified"].between(0, 1).all()

