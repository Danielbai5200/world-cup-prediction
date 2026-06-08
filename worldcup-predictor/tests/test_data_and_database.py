from __future__ import annotations

import pandas as pd

from src.ingestion.data_sources import CsvSampleDataSource
from src.ingestion.database import create_tables, load_sample_data, read_table


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

