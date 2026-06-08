from __future__ import annotations

import sys
from time import perf_counter
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


TEAM_NAME_ZH = {
    "Argentina": "阿根廷",
    "France": "法国",
    "England": "英格兰",
    "Spain": "西班牙",
    "Germany": "德国",
    "Brazil": "巴西",
    "Portugal": "葡萄牙",
    "Netherlands": "荷兰",
    "Japan": "日本",
    "United States": "美国",
}

PLAYER_NAME_ZH = {
    "Lionel Messi": "利昂内尔·梅西",
    "Julian Alvarez": "胡利安·阿尔瓦雷斯",
    "Kylian Mbappe": "基利安·姆巴佩",
    "Antoine Griezmann": "安托万·格列兹曼",
    "Harry Kane": "哈里·凯恩",
    "Jude Bellingham": "裘德·贝林厄姆",
    "Lamine Yamal": "拉明·亚马尔",
    "Pedri": "佩德里",
    "Florian Wirtz": "弗洛里安·维尔茨",
    "Jamal Musiala": "贾马尔·穆西亚拉",
    "Vinicius Junior": "维尼修斯",
    "Rodrygo": "罗德里戈",
    "Bruno Fernandes": "布鲁诺·费尔南德斯",
    "Bernardo Silva": "贝尔纳多·席尔瓦",
    "Virgil van Dijk": "维吉尔·范戴克",
    "Xavi Simons": "哈维·西蒙斯",
    "Takefusa Kubo": "久保建英",
    "Kaoru Mitoma": "三笘薰",
    "Christian Pulisic": "克里斯蒂安·普利西奇",
    "Tyler Adams": "泰勒·亚当斯",
}

POSITION_ZH = {
    "FW": "前锋",
    "MF": "中场",
    "DF": "后卫",
    "GK": "门将",
}

INJURY_STATUS_ZH = {
    "fit": "健康",
    "minor": "轻伤",
    "doubtful": "出战存疑",
    "injured": "受伤",
    "out": "缺阵",
}

COLUMN_LABELS_ZH = {
    "name": "球队",
    "team": "球队",
    "fifa_rank": "FIFA排名",
    "elo_rating": "Elo评分",
    "market_value": "阵容身价",
    "attack_rating": "进攻评分",
    "defense_rating": "防守评分",
    "overall_rating": "综合评分",
    "updated_at": "更新时间",
    "position": "位置",
    "age": "年龄",
    "form_score": "状态评分",
    "fitness_score": "健康评分",
    "injury_status": "伤病状态",
    "player_name": "球员",
    "severity": "严重程度",
    "expected_return": "预计回归",
    "score": "比分",
    "probability_pct": "概率 %",
    "group_qualified": "小组出线 %",
    "round_32": "32强 %",
    "round_16": "16强 %",
    "quarterfinal": "8强 %",
    "semifinal": "4强 %",
    "final": "决赛 %",
    "champion": "夺冠 %",
}

TEAM_COLUMN_LABELS_ZH = {**COLUMN_LABELS_ZH, "name": "球队"}
PLAYER_COLUMN_LABELS_ZH = {**COLUMN_LABELS_ZH, "name": "球员", "team": "球队"}
INJURY_COLUMN_LABELS_ZH = {**COLUMN_LABELS_ZH, "team": "球队"}


st.set_page_config(page_title="2026世界杯预测系统", layout="wide", initial_sidebar_state="expanded")


st.markdown(
    """
    <style>
    .stDeployButton {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #stDecoration {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def team_label(team: str) -> str:
    if team.startswith("Qualifier "):
        return f"资格赛球队 {team.split()[-1]}"
    return TEAM_NAME_ZH.get(team, team)


def localize_team_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "name" in result.columns:
        result["name"] = result["name"].map(lambda value: team_label(str(value)) if pd.notna(value) else "")
    if "team" in result.columns:
        result["team"] = result["team"].map(lambda value: team_label(str(value)) if pd.notna(value) else "")
    return result


def localize_player_values(df: pd.DataFrame) -> pd.DataFrame:
    result = localize_team_values(df)
    if "name" in result.columns:
        result["name"] = result["name"].map(lambda value: PLAYER_NAME_ZH.get(str(value), value) if pd.notna(value) else "")
    if "position" in result.columns:
        result["position"] = result["position"].map(lambda value: POSITION_ZH.get(str(value), value) if pd.notna(value) else "")
    if "injury_status" in result.columns:
        result["injury_status"] = result["injury_status"].map(lambda value: INJURY_STATUS_ZH.get(str(value), value) if pd.notna(value) else "")
    return result


def localize_injury_values(df: pd.DataFrame) -> pd.DataFrame:
    result = localize_team_values(df)
    if "player_name" in result.columns:
        result["player_name"] = result["player_name"].map(lambda value: PLAYER_NAME_ZH.get(str(value), value) if pd.notna(value) else "")
    if "injury_status" in result.columns:
        result["injury_status"] = result["injury_status"].map(lambda value: INJURY_STATUS_ZH.get(str(value), value) if pd.notna(value) else "")
    if "expected_return" in result.columns:
        result["expected_return"] = result["expected_return"].fillna("").astype(str).replace({"nan": ""})
    return result


def with_chinese_team_names(df: pd.DataFrame, column: str = "team") -> pd.DataFrame:
    result = df.copy()
    if column in result.columns:
        result[column] = result[column].map(team_label)
    if "name" in result.columns:
        result["name"] = result["name"].map(team_label)
    return result


def rename_columns_zh(df: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    label_map = labels or COLUMN_LABELS_ZH
    return df.rename(columns={column: label_map.get(column, column) for column in df.columns})


def single_match_page() -> None:
    predictor = get_predictor()
    teams = predictor.team_names
    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("主队", teams, index=teams.index("Argentina") if "Argentina" in teams else 0, format_func=team_label)
    with col_away:
        default_away = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
        away_team = st.selectbox("客队", teams, index=default_away, format_func=team_label)
    if home_team == away_team:
        st.warning("请选择两支不同球队。")
        return
    prediction = predictor.predict_match(home_team, away_team)
    probs = prediction["probabilities"]
    xg = prediction["expected_goals"]
    metrics = st.columns(5)
    metrics[0].metric("主胜概率", format_probability(probs["home_win"]))
    metrics[1].metric("平局概率", format_probability(probs["draw"]))
    metrics[2].metric("客胜概率", format_probability(probs["away_win"]))
    metrics[3].metric("主队预期进球", f"{xg['home']:.2f}")
    metrics[4].metric("客队预期进球", f"{xg['away']:.2f}")
    score_df = pd.DataFrame(prediction["top_scores"])
    score_df["probability_pct"] = score_df["probability"].map(lambda value: 100 * value)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("最可能比分前10")
        st.dataframe(
            rename_columns_zh(score_df[["score", "probability_pct"]]),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        matrix = prediction["score_matrix"]
        heatmap_df = matrix.reset_index().melt(id_vars="index", var_name="客队进球", value_name="概率")
        heatmap_df = heatmap_df.rename(columns={"index": "主队进球"})
        fig = px.density_heatmap(
            heatmap_df,
            x="客队进球",
            y="主队进球",
            z="概率",
            text_auto=".2%",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def simulation_page() -> None:
    n_simulations = st.slider("模拟次数", min_value=1_000, max_value=100_000, value=10_000, step=1_000)
    start = st.button("开始模拟", type="primary")
    if start:
        with st.spinner(f"正在运行 {n_simulations:,} 次世界杯模拟，请稍候..."):
            started_at = perf_counter()
            st.session_state["simulation_result"] = run_simulation(n_simulations)
            st.session_state["simulation_count"] = n_simulations
            st.session_state["simulation_elapsed"] = perf_counter() - started_at
        st.success(f"模拟完成：共运行 {n_simulations:,} 次，用时 {st.session_state['simulation_elapsed']:.2f} 秒。")
    if "simulation_result" not in st.session_state:
        st.info("请选择模拟次数，然后点击“开始模拟”。模拟次数越高，结果越稳定，但等待时间也越长。")
        return
    result = st.session_state["simulation_result"]
    st.caption(
        f"当前展示结果来自 {st.session_state.get('simulation_count', n_simulations):,} 次模拟，"
        f"耗时 {st.session_state.get('simulation_elapsed', 0):.2f} 秒。"
    )
    col_a, col_b = st.columns(2)
    champion = result.head(12).copy()
    champion["team"] = champion["team"].map(team_label)
    champion["夺冠概率"] = champion["champion"] * 100
    with col_a:
        st.subheader("夺冠概率排行榜")
        fig = px.bar(
            champion,
            x="夺冠概率",
            y="team",
            orientation="h",
            color="夺冠概率",
            labels={"team": "球队"},
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    advancement = result.head(16).copy()
    for col in ["group_qualified", "round_32", "round_16", "quarterfinal", "semifinal", "final", "champion"]:
        advancement[col] *= 100
    advancement = with_chinese_team_names(advancement)
    with col_b:
        st.subheader("晋级概率排行榜")
        st.dataframe(
            rename_columns_zh(advancement),
            use_container_width=True,
            hide_index=True,
        )


def data_center_page() -> None:
    data = load_data()
    teams, players, injuries = data["teams"], data["players"], data["injuries"]
    tab_teams, tab_players, tab_injuries = st.tabs(["球队评分", "球员状态", "伤病情况"])
    with tab_teams:
        st.dataframe(rename_columns_zh(localize_team_values(teams.sort_values("overall_rating", ascending=False)), TEAM_COLUMN_LABELS_ZH), use_container_width=True, hide_index=True)
        chart_teams = localize_team_values(teams)
        fig = px.scatter(
            chart_teams,
            x="attack_rating",
            y="defense_rating",
            size="market_value",
            color="overall_rating",
            hover_name="name",
            labels={
                "attack_rating": "进攻评分",
                "defense_rating": "防守评分",
                "market_value": "阵容身价",
                "overall_rating": "综合评分",
                "name": "球队",
            },
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with tab_players:
        st.dataframe(rename_columns_zh(localize_player_values(players.sort_values(["team", "form_score"], ascending=[True, False])), PLAYER_COLUMN_LABELS_ZH), use_container_width=True, hide_index=True)
    with tab_injuries:
        st.dataframe(rename_columns_zh(localize_injury_values(injuries), INJURY_COLUMN_LABELS_ZH), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("2026世界杯预测系统")
    st.sidebar.title("功能选择")
    page = st.sidebar.radio("页面", ["单场预测", "世界杯模拟", "数据中心"])
    if page == "单场预测":
        single_match_page()
    elif page == "世界杯模拟":
        simulation_page()
    else:
        data_center_page()


if __name__ == "__main__":
    main()
