from __future__ import annotations

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    fifa_rank = Column(Integer)
    elo_rating = Column(Float)
    market_value = Column(Float)
    attack_rating = Column(Float)
    defense_rating = Column(Float)
    overall_rating = Column(Float)
    updated_at = Column(Date)


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    team = Column(String, nullable=False)
    position = Column(String)
    age = Column(Integer)
    market_value = Column(Float)
    form_score = Column(Float)
    fitness_score = Column(Float)
    injury_status = Column(String)
    updated_at = Column(Date)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    xg_home = Column(Float)
    xg_away = Column(Float)
    competition = Column(String)


class Odd(Base):
    __tablename__ = "odds"

    match_id = Column(Integer, ForeignKey("matches.id"), primary_key=True)
    home_win_odds = Column(Float)
    draw_odds = Column(Float)
    away_win_odds = Column(Float)
    timestamp = Column(DateTime)

