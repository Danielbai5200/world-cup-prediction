from __future__ import annotations

import pandas as pd

from src.ingestion.data_sources import CsvSampleDataSource
from src.ingestion.database import create_tables, load_sample_data, read_table
from src.ingestion.daily_update import run_daily_update
from src.ingestion.odds_sources import (
    apply_match_odds_snapshot,
    devig_probabilities,
    expected_value,
    parse_the_odds_api_match_odds,
    update_match_odds_data,
    update_prediction_market_odds,
)
from src.ingestion.real_data_update import apply_elo_rankings, parse_elo_rankings, parse_fifa_ranking_metadata, update_elo_data
from src.ingestion.source_mapping import discover_onefootball_team, load_team_source_mapping, load_world_cup_teams, validate_team_source_mapping
from src.ingestion.transfermarkt import apply_player_snapshot, parse_transfermarkt_squad


def test_csv_sample_data_loads_required_entities() -> None:
    source = CsvSampleDataSource()
    teams = source.teams()
    assert {"Argentina", "France", "Brazil"}.issubset(set(teams["name"]))
    assert len(source.players()) >= 20
    assert len(source.matches()) >= 10
    assert len(source.odds()) >= 10


def test_database_initialization_loads_tables(tmp_path) -> None:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    teams = read_table("teams", engine)
    matches = read_table("matches", engine)
    assert isinstance(teams, pd.DataFrame)
    assert len(teams) == 10
    assert len(matches) == 15


def test_parse_and_apply_real_elo_rankings(tmp_path) -> None:
    from sqlalchemy import create_engine

    html = """
    <h2 class="titre-paragraphe">World football Elo ratings<em>as on June 6th, 2026</em></h2>
    <tr><th>1.</th><td><span class="opensans">Spain</span></td><td><font>2155</font></td></tr>
    <tr><th>2.</th><td><span class="opensans">Argentina</span></td><td><font>2114</font></td></tr>
    """
    rankings = parse_elo_rankings(html)
    assert rankings.loc[0, "team"] == "Spain"
    assert rankings.loc[0, "elo_rating"] == 2155

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    rows_updated = apply_elo_rankings(rankings, engine)
    teams = read_table("teams", engine)
    argentina = teams.loc[teams["name"] == "Argentina"].iloc[0]
    assert rows_updated == 2
    assert argentina["elo_rating"] == 2114
    assert argentina["fifa_rank"] == 2


def test_update_elo_data_writes_raw_processed_and_database(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine
    import src.ingestion.real_data_update as real_update

    html = """
    <h2 class="titre-paragraphe">World football Elo ratings<em>as on June 6th, 2026</em></h2>
    <tr><th>1.</th><td><span class="opensans">Spain</span></td><td><font>2155</font></td></tr>
    <tr><th>2.</th><td><span class="opensans">Argentina</span></td><td><font>2114</font></td></tr>
    """
    monkeypatch.setattr(real_update, "fetch_text", lambda: html)
    monkeypatch.setattr(
        real_update,
        "_safe_fetch_fifa_metadata",
        lambda: {"fifa_last_official_update": "01 April 2026", "fifa_next_official_update": "11 June 2026"},
    )
    monkeypatch.setattr(real_update, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(real_update, "PROCESSED_DATA_DIR", tmp_path / "processed")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    result = update_elo_data(engine)
    assert result.rows_downloaded == 2
    assert result.rows_updated == 2
    assert result.metadata["fifa_last_official_update"] == "01 April 2026"
    assert result.raw_path.exists()
    assert result.processed_path.exists()


def test_daily_update_reports_all_sections(monkeypatch) -> None:
    import src.ingestion.daily_update as daily_update

    monkeypatch.setattr(daily_update, "initialize_database", lambda: None)
    mapping = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "fifa_name": "Argentina",
                "fbref_squad_id": "f9fddd6e",
                "fbref_slug": "Argentina",
                "confederation": "CONMEBOL",
                "world_cup_group": "C",
                "fbref_stats_url": "https://fbref.com/en/squads/f9fddd6e/Argentina-Men-Stats",
                "fbref_history_url": "https://fbref.com/en/squads/f9fddd6e/history/Argentina-Men-Stats-and-History",
                "onefootball_url": "https://onefootball.com/en/team/argentina-55",
                "onefootball_status": "verified_http_200",
                "transfermarkt_slug": "argentinien",
                "transfermarkt_team_id": "3437",
                "transfermarkt_url": "https://www.transfermarkt.com/argentinien/startseite/verein/3437",
            }
        ]
    )
    monkeypatch.setattr(daily_update, "load_team_source_mapping", lambda: mapping)
    monkeypatch.setattr(
        daily_update,
        "update_elo_data",
        lambda: type(
            "Result",
            (),
            {
                "source": "test-source",
                "rows_downloaded": 2,
                "rows_updated": 1,
                "raw_path": "raw.html",
                "processed_path": "processed.csv",
                "metadata": {"fifa_last_official_update": "01 April 2026"},
            },
        )(),
    )
    monkeypatch.setattr(
        daily_update,
        "update_transfermarkt_player_data",
        lambda max_teams=None, request_delay_seconds=3.0: type(
            "Result",
            (),
            {
                "source": "Transfermarkt",
                "teams_requested": 1,
                "teams_updated": 1,
                "rows_downloaded": 2,
                "rows_updated": 2,
                "raw_dir": "raw/transfermarkt",
                "processed_path": "processed/players.csv",
                "metadata": {"failures": {}},
            },
        )(),
    )
    monkeypatch.setattr(
        daily_update,
        "update_match_odds_data",
        lambda: type(
            "Result",
            (),
            {
                "source": "The Odds API",
                "rows_downloaded": 0,
                "rows_updated": 0,
                "processed_path": "processed/odds.csv",
                "metadata": {"status": "skipped"},
            },
        )(),
    )
    monkeypatch.setattr(
        daily_update,
        "update_prediction_market_odds",
        lambda: type(
            "Result",
            (),
            {
                "source": "Polymarket",
                "rows_downloaded": 0,
                "rows_updated": 0,
                "processed_path": "processed/winner.csv",
                "metadata": {"status": "skipped"},
            },
        )(),
    )
    result = run_daily_update()
    assert result["status"] == "ok"
    assert result["team_update"]["rows_updated"] == 1
    assert result["team_update"]["metadata"]["fifa_last_official_update"] == "01 April 2026"
    assert result["player_update"]["status"] == "ok"
    assert result["odds_update"]["status"] == "skipped"
    assert result["source_mapping"]["fbref_urls"] == 1
    assert result["source_mapping"]["transfermarkt_urls"] == 1
    assert result["source_mapping"]["missing_fbref"] == []


def test_parse_fifa_ranking_metadata() -> None:
    html = """
    <div>Last official update:</div><span>01 April 2026</span>
    <div>Next official update:</div><span>11 June 2026 (4 days)</span>
    """
    metadata = parse_fifa_ranking_metadata(html)
    assert metadata["fifa_last_official_update"] == "01 April 2026"
    assert metadata["fifa_next_official_update"] == "11 June 2026 (4 days)"


def test_team_source_mapping_covers_sample_teams() -> None:
    mapping = load_team_source_mapping()
    validation = validate_team_source_mapping(mapping)
    assert validation.teams == 48
    assert validation.fbref_urls >= 10
    assert validation.onefootball_urls == 48
    assert validation.transfermarkt_urls == 48
    assert {"Argentina", "France", "Brazil", "Uzbekistan"}.issubset(set(mapping["team"]))
    assert validation.missing_onefootball == []
    assert validation.missing_transfermarkt == []


def test_world_cup_team_list_contains_48_teams() -> None:
    teams = load_world_cup_teams()
    assert len(teams) == 48
    assert set(teams["group"]) == set("ABCDEFGHIJKL")
    assert {"Canada", "Mexico", "United States"}.issubset(set(teams["team"]))


def test_discover_onefootball_team_from_search_api(monkeypatch) -> None:
    import src.ingestion.source_mapping as source_mapping

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"teams":[{"id":61,"name":"England","is_national":true,"url":"/en/team/england-61"}]}'

    monkeypatch.setattr(source_mapping, "urlopen", lambda request, timeout: Response())
    result = discover_onefootball_team("England")
    assert result is not None
    assert result["onefootball_team_id"] == "61"
    assert result["onefootball_url"] == "https://onefootball.com/en/team/england-61"


def test_parse_transfermarkt_squad_extracts_players() -> None:
    html = """
    <table class="items"><tbody>
    <tr class="odd">
      <td><table class="inline-table"><tr><td class="hauptlink"><a>Lionel Messi</a></td></tr><tr><td>Right Winger</td></tr></table></td>
      <td>Jun 24, 1987 (38)</td><td class="rechts hauptlink">€15.00m</td>
    </tr>
    <tr class="even">
      <td><table class="inline-table"><tr><td class="hauptlink"><a>Emiliano Martinez</a></td></tr><tr><td>Goalkeeper</td></tr></table></td>
      <td>Sep 2, 1992 (33)</td><td><span title="Injury">injury</span></td><td class="rechts hauptlink">€20.00m</td>
    </tr>
    </tbody></table>
    """
    players = parse_transfermarkt_squad(html, "Argentina")
    assert len(players) == 2
    assert players.loc[0, "position"] == "FW"
    assert players.loc[0, "market_value"] == 15
    assert players.loc[1, "position"] == "GK"
    assert players.loc[1, "injury_status"] == "injured"


def test_parse_the_odds_api_match_odds_and_devig() -> None:
    data = [
        {
            "id": "event-1",
            "commence_time": "2026-06-11T20:00:00Z",
            "home_team": "Argentina",
            "away_team": "France",
            "bookmakers": [
                {
                    "key": "testbook",
                    "last_update": "2026-06-08T00:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Argentina", "price": 2.2},
                                {"name": "Draw", "price": 3.2},
                                {"name": "France", "price": 3.0},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    odds = parse_the_odds_api_match_odds(data)
    probs = devig_probabilities([2.2, 3.2, 3.0])
    assert len(odds) == 1
    assert odds.loc[0, "home_team"] == "Argentina"
    assert round(odds.loc[0, "home_win_prob_market"], 8) == round(float(probs[0]), 8)
    assert expected_value(0.5, 2.2) > 0


def test_apply_player_snapshot_replaces_team_players(tmp_path) -> None:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    players = pd.DataFrame(
        [
            {
                "name": "New Player",
                "team": "Argentina",
                "position": "FW",
                "age": 24,
                "market_value": 12.5,
                "form_score": 70.0,
                "fitness_score": 90.0,
                "injury_status": "fit",
                "updated_at": pd.Timestamp("2026-06-08"),
            }
        ]
    )
    rows = apply_player_snapshot(players, engine)
    stored = read_table("players", engine)
    argentina_players = stored.loc[stored["team"] == "Argentina"]
    assert rows == 1
    assert argentina_players["name"].tolist() == ["New Player"]


def test_apply_match_odds_snapshot_updates_matching_fixture(tmp_path) -> None:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    odds = pd.DataFrame(
        [
                {
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "home_win_odds": 2.1,
                    "draw_odds": 3.3,
                    "away_win_odds": 3.1,
                "timestamp": "2026-06-08T00:00:00Z",
            }
        ]
    )
    rows = apply_match_odds_snapshot(odds, engine)
    stored = read_table("odds", engine)
    assert rows == 1
    assert 2.1 in set(stored["home_win_odds"])


def test_odds_updates_skip_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_SLUG", raising=False)
    match_result = update_match_odds_data()
    market_result = update_prediction_market_odds()
    assert match_result.rows_updated == 0
    assert match_result.metadata["status"] == "skipped"
    assert market_result.rows_updated == 0
    assert market_result.metadata["status"] == "skipped"


def test_update_match_odds_data_fetches_and_applies(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine
    import src.ingestion.odds_sources as odds_sources

    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(odds_sources, "DATABASE_PATH", tmp_path / "test.sqlite")
    monkeypatch.setattr(odds_sources, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(odds_sources, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        odds_sources,
        "fetch_json",
        lambda url: [
            {
                "id": "event-1",
                "home_team": "Argentina",
                "away_team": "Brazil",
                "bookmakers": [
                    {
                        "key": "testbook",
                        "last_update": "2026-06-08T00:00:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Argentina", "price": 2.1},
                                    {"name": "Draw", "price": 3.3},
                                    {"name": "Brazil", "price": 3.1},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    )
    result = update_match_odds_data(engine)
    assert result.rows_downloaded == 1
    assert result.rows_updated == 1
    assert result.raw_path.exists()
    assert result.processed_path.exists()


def test_update_prediction_market_odds_fetches_polymarket(tmp_path, monkeypatch) -> None:
    import src.ingestion.odds_sources as odds_sources

    monkeypatch.setenv("POLYMARKET_SLUG", "world-cup-2026-winner")
    monkeypatch.setattr(odds_sources, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        odds_sources,
        "fetch_json",
        lambda url: [{"outcomes": '["Argentina","France"]', "outcomePrices": '["0.25","0.20"]'}],
    )
    result = update_prediction_market_odds()
    assert result.rows_downloaded == 2
    assert result.rows_updated == 2
    assert result.processed_path.exists()


def test_update_transfermarkt_player_data_fetches_and_applies(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine
    import src.ingestion.transfermarkt as transfermarkt

    html = """
    <table class="items"><tbody>
    <tr class="odd">
      <td><table class="inline-table"><tr><td class="hauptlink"><a>Lionel Messi</a></td></tr><tr><td>Right Winger</td></tr></table></td>
      <td>Jun 24, 1987 (38)</td><td class="rechts hauptlink">€15.00m</td>
    </tr>
    </tbody></table>
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", future=True)
    create_tables(engine)
    load_sample_data(engine)
    mapping = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "fifa_name": "Argentina",
                "confederation": "CONMEBOL",
                "world_cup_group": "C",
                "fbref_squad_id": "",
                "fbref_slug": "",
                "fbref_stats_url": "",
                "fbref_history_url": "",
                "onefootball_url": "https://onefootball.com/en/team/argentina-55",
                "onefootball_status": "verified_http_200",
                "transfermarkt_slug": "argentinien",
                "transfermarkt_team_id": "3437",
                "transfermarkt_url": "https://www.transfermarkt.com/argentinien/startseite/verein/3437",
            }
        ]
    )
    monkeypatch.setattr(transfermarkt, "DATABASE_PATH", tmp_path / "test.sqlite")
    monkeypatch.setattr(transfermarkt, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(transfermarkt, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(transfermarkt, "load_team_source_mapping", lambda: mapping)
    monkeypatch.setattr(transfermarkt, "fetch_transfermarkt_html", lambda url: html)
    result = transfermarkt.update_transfermarkt_player_data(engine, request_delay_seconds=0)
    assert result.teams_updated == 1
    assert result.rows_updated == 1
    assert result.processed_path.exists()
