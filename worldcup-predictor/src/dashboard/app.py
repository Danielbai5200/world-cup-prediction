from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.data_sources import CsvSampleDataSource
from src.models.predictor import MatchPredictor
from src.simulation import WorldCupSimulator
from src.utils.config import SAMPLE_DATA_DIR


st.set_page_config(page_title="World Cup Predictor 2026", layout="wide")


@st.cache_resource
def get_predictor() -> MatchPredictor:
    return MatchPredictor()


@st.cache_data(show_spinner=False)
def run_simulation(n_simulations: int) -> pd.DataFrame:
    simulator = WorldCupSimulator(get_predictor(), random_seed=2026)
    return simulator.simulate(n_simulations=n_simulations)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    source = CsvSampleDataSource(SAMPLE_DATA_DIR)
    return {
        "teams": source.teams(),
        "players": source.players(),
        "matches": source.matches(),
        "odds": source.odds(),
        "injuries": pd.read_csv(SAMPLE_DATA_DIR / "injuries.csv"),
    }


def format_probability(value: float) -> str:
    return f"{100 * value:.1f}%"


def single_match_page() -> None:
    predictor = get_predictor()
    teams = predictor.team_names
    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("Home Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col_away:
        default_away = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
        away_team = st.selectbox("Away Team", teams, index=default_away)
    if home_team == away_team:
        st.warning("Choose two different teams.")
        return
    prediction = predictor.predict_match(home_team, away_team)
    probs = prediction["probabilities"]
    xg = prediction["expected_goals"]
    metrics = st.columns(5)
    metrics[0].metric("Home Win", format_probability(probs["home_win"]))
    metrics[1].metric("Draw", format_probability(probs["draw"]))
    metrics[2].metric("Away Win", format_probability(probs["away_win"]))
    metrics[3].metric("Home xG", f"{xg['home']:.2f}")
    metrics[4].metric("Away xG", f"{xg['away']:.2f}")
    score_df = pd.DataFrame(prediction["top_scores"])
    score_df["probability_pct"] = score_df["probability"].map(lambda value: 100 * value)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Top 10 Scorelines")
        st.dataframe(
            score_df[["score", "probability_pct"]].rename(columns={"score": "Score", "probability_pct": "Probability %"}),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        matrix = prediction["score_matrix"]
        heatmap_df = matrix.reset_index().melt(id_vars="index", var_name="Away Goals", value_name="Probability")
        heatmap_df = heatmap_df.rename(columns={"index": "Home Goals"})
        fig = px.density_heatmap(
            heatmap_df,
            x="Away Goals",
            y="Home Goals",
            z="Probability",
            text_auto=".2%",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)


def simulation_page() -> None:
    n_simulations = st.slider("Simulations", min_value=1_000, max_value=100_000, value=10_000, step=1_000)
    result = run_simulation(n_simulations)
    col_a, col_b = st.columns(2)
    champion = result.head(12).copy()
    champion["Champion Probability"] = champion["champion"] * 100
    with col_a:
        st.subheader("Champion Probability")
        fig = px.bar(champion, x="Champion Probability", y="team", orientation="h", color="Champion Probability", color_continuous_scale="Viridis")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    advancement = result.head(16).copy()
    for col in ["group_qualified", "round_32", "round_16", "quarterfinal", "semifinal", "final", "champion"]:
        advancement[col] *= 100
    with col_b:
        st.subheader("Advancement Rankings")
        st.dataframe(
            advancement.rename(
                columns={
                    "team": "Team",
                    "group_qualified": "Group Qual %",
                    "round_32": "Round 32 %",
                    "round_16": "Round 16 %",
                    "quarterfinal": "Quarterfinal %",
                    "semifinal": "Semifinal %",
                    "final": "Final %",
                    "champion": "Champion %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def data_center_page() -> None:
    data = load_data()
    teams, players, injuries = data["teams"], data["players"], data["injuries"]
    tab_teams, tab_players, tab_injuries = st.tabs(["Teams", "Players", "Injuries"])
    with tab_teams:
        st.dataframe(teams.sort_values("overall_rating", ascending=False), use_container_width=True, hide_index=True)
        fig = px.scatter(
            teams,
            x="attack_rating",
            y="defense_rating",
            size="market_value",
            color="overall_rating",
            hover_name="name",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab_players:
        st.dataframe(players.sort_values(["team", "form_score"], ascending=[True, False]), use_container_width=True, hide_index=True)
    with tab_injuries:
        st.dataframe(injuries, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("World Cup Predictor 2026")
    page = st.sidebar.radio("Page", ["Single Match", "World Cup Simulation", "Data Center"])
    if page == "Single Match":
        single_match_page()
    elif page == "World Cup Simulation":
        simulation_page()
    else:
        data_center_page()


if __name__ == "__main__":
    main()
