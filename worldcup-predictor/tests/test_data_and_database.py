from __future__ import annotations

import pandas as pd

from src.ingestion.data_sources import CsvSampleDataSource
from src.ingestion.database import create_tables, load_sample_data, read_table
from src.ingestion.daily_update import run_daily_update
from src.ingestion.real_data_update import apply_elo_rankings, parse_elo_rankings, parse_fifa_ranking_metadata, update_elo_data


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
    result = run_daily_update()
    assert result["status"] == "ok"
    assert result["team_update"]["rows_updated"] == 1
    assert result["team_update"]["metadata"]["fifa_last_official_update"] == "01 April 2026"
    assert result["player_update"]["status"] == "skipped"
    assert result["odds_update"]["status"] == "skipped"


def test_parse_fifa_ranking_metadata() -> None:
    html = """
    <div>Last official update:</div><span>01 April 2026</span>
    <div>Next official update:</div><span>11 June 2026 (4 days)</span>
    """
    metadata = parse_fifa_ranking_metadata(html)
    assert metadata["fifa_last_official_update"] == "01 April 2026"
    assert metadata["fifa_next_official_update"] == "11 June 2026 (4 days)"
