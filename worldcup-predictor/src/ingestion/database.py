from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.ingestion.schema import Base
from src.utils.config import DATABASE_PATH, DATABASE_URL, SAMPLE_DATA_DIR


def get_engine(database_url: str = DATABASE_URL) -> Engine:
    return create_engine(database_url, future=True)


def create_tables(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def load_sample_data(engine: Engine | None = None, sample_dir: Path = SAMPLE_DATA_DIR) -> None:
    engine = engine or get_engine()
    create_tables(engine)
    loaders = {
        "teams": ("teams.csv", ["updated_at"]),
        "players": ("players.csv", ["updated_at"]),
        "matches": ("matches.csv", ["date"]),
        "odds": ("odds.csv", ["timestamp"]),
    }
    with engine.begin() as conn:
        for table in loaders:
            conn.execute(text(f"DELETE FROM {table}"))
        for table, (filename, date_columns) in loaders.items():
            df = pd.read_csv(sample_dir / filename, parse_dates=date_columns)
            for column in date_columns:
                if column != "timestamp":
                    df[column] = df[column].dt.date
            df.to_sql(table, conn, if_exists="append", index=False)


def initialize_database(database_path: Path = DATABASE_PATH) -> Path:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(f"sqlite:///{database_path}")
    load_sample_data(engine)
    return database_path


def read_table(table: str, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql_table(table, engine)
