from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion.database import get_engine, initialize_database
from src.utils.config import DATABASE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
USER_AGENT = "WorldCupPredictor2026/1.0 (+local odds updater)"


@dataclass(frozen=True)
class OddsUpdateResult:
    source: str
    raw_path: Path
    processed_path: Path
    rows_downloaded: int
    rows_updated: int
    updated_at: str
    metadata: dict[str, object]


def devig_probabilities(decimal_odds: list[float] | np.ndarray) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if np.any(odds <= 1):
        raise ValueError("Decimal odds must be greater than 1.")
    implied = 1 / odds
    return implied / implied.sum()


def expected_value(probability: float, decimal_odds: float) -> float:
    return probability * (decimal_odds - 1) - (1 - probability)


def fetch_json(url: str, timeout: int = 30) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_the_odds_api_match_odds(
    api_key: str,
    sport_key: str = "soccer_fifa_world_cup",
    regions: str = "us,uk,eu",
    markets: str = "h2h",
) -> pd.DataFrame:
    query = urlencode(
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
    )
    url = f"{THE_ODDS_API_BASE_URL}/{sport_key}/odds?{query}"
    data = fetch_json(url)
    return parse_the_odds_api_match_odds(data)


def parse_the_odds_api_match_odds(data: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not isinstance(data, list):
        return pd.DataFrame(rows)
    for event in data:
        if not isinstance(event, dict):
            continue
        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        if not home_team or not away_team:
            continue
        for bookmaker in event.get("bookmakers", []) or []:
            book = str(bookmaker.get("key") or bookmaker.get("title") or "")
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                outcome_prices = {str(item.get("name")): float(item.get("price")) for item in market.get("outcomes", [])}
                home_odds = outcome_prices.get(home_team)
                away_odds = outcome_prices.get(away_team)
                draw_odds = outcome_prices.get("Draw")
                if not home_odds or not away_odds or not draw_odds:
                    continue
                probs = devig_probabilities([home_odds, draw_odds, away_odds])
                rows.append(
                    {
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": book,
                        "home_win_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_win_odds": away_odds,
                        "home_win_prob_market": probs[0],
                        "draw_prob_market": probs[1],
                        "away_win_prob_market": probs[2],
                        "timestamp": market.get("last_update") or bookmaker.get("last_update") or datetime.now(timezone.utc).isoformat(),
                    }
                )
    return pd.DataFrame(rows)


def update_match_odds_data(engine: Engine | None = None) -> OddsUpdateResult:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        return _skipped_result("The Odds API", "ODDS_API_KEY is not configured.")

    engine = engine or get_engine()
    if not DATABASE_PATH.exists():
        initialize_database()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc)
    sport_key = os.getenv("ODDS_API_SPORT_KEY", "soccer_fifa_world_cup")
    raw_data = fetch_json(
        f"{THE_ODDS_API_BASE_URL}/{sport_key}/odds?"
        + urlencode(
            {
                "apiKey": api_key,
                "regions": os.getenv("ODDS_API_REGIONS", "us,uk,eu"),
                "markets": "h2h",
                "oddsFormat": "decimal",
            }
        )
    )
    raw_path = RAW_DATA_DIR / f"the_odds_api_{fetched_at:%Y%m%d%H%M%S}.json"
    raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")
    odds = parse_the_odds_api_match_odds(raw_data)
    processed_path = PROCESSED_DATA_DIR / "match_odds_latest.csv"
    odds.to_csv(processed_path, index=False)
    rows_updated = apply_match_odds_snapshot(odds, engine) if not odds.empty else 0
    return OddsUpdateResult(
        source="The Odds API",
        raw_path=raw_path,
        processed_path=processed_path,
        rows_downloaded=len(odds),
        rows_updated=rows_updated,
        updated_at=fetched_at.isoformat(),
        metadata={"sport_key": sport_key},
    )


def apply_match_odds_snapshot(odds: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    rows_updated = 0
    with engine.begin() as conn:
        matches = pd.read_sql("SELECT id, home_team, away_team FROM matches", conn)
        match_key_to_id = {(row.home_team, row.away_team): int(row.id) for row in matches.itertuples()}
        latest = odds.sort_values("timestamp").groupby(["home_team", "away_team"], as_index=False).tail(1)
        for row in latest.itertuples(index=False):
            match_id = match_key_to_id.get((row.home_team, row.away_team))
            if match_id is None:
                continue
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO odds (
                        match_id, home_win_odds, draw_odds, away_win_odds, timestamp
                    ) VALUES (
                        :match_id, :home_win_odds, :draw_odds, :away_win_odds, :timestamp
                    )
                    """
                ),
                {
                    "match_id": match_id,
                    "home_win_odds": float(row.home_win_odds),
                    "draw_odds": float(row.draw_odds),
                    "away_win_odds": float(row.away_win_odds),
                    "timestamp": row.timestamp,
                },
            )
            rows_updated += 1
    return rows_updated


def fetch_polymarket_winner_odds(slug: str) -> pd.DataFrame:
    data = fetch_json(f"{POLYMARKET_GAMMA_URL}?{urlencode({'slug': slug})}")
    rows: list[dict[str, object]] = []
    if not isinstance(data, list):
        return pd.DataFrame(rows)
    for market in data:
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            prices = [float(value) for value in json.loads(market.get("outcomePrices", "[]"))]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not outcomes or len(outcomes) != len(prices):
            continue
        price_array = np.asarray(prices, dtype=float)
        fair = price_array / price_array.sum() if price_array.sum() > 0 else np.zeros(len(price_array))
        for outcome, price, fair_prob in zip(outcomes, prices, fair):
            if price <= 0:
                continue
            rows.append(
                {
                    "market": "winner",
                    "team": outcome,
                    "bookmaker": "polymarket",
                    "decimal_odds": 1 / price,
                    "implied_prob": price,
                    "fair_prob": fair_prob,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def update_prediction_market_odds() -> OddsUpdateResult:
    slug = os.getenv("POLYMARKET_SLUG", "").strip()
    if not slug:
        return _skipped_result("Polymarket", "POLYMARKET_SLUG is not configured.")
    fetched_at = datetime.now(timezone.utc)
    odds = fetch_polymarket_winner_odds(slug)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "winner_market_odds_latest.csv"
    odds.to_csv(processed_path, index=False)
    return OddsUpdateResult(
        source="Polymarket",
        raw_path=processed_path,
        processed_path=processed_path,
        rows_downloaded=len(odds),
        rows_updated=len(odds),
        updated_at=fetched_at.isoformat(),
        metadata={"slug": slug},
    )


def _skipped_result(source: str, reason: str) -> OddsUpdateResult:
    path = PROCESSED_DATA_DIR / "odds_skipped.csv"
    return OddsUpdateResult(
        source=source,
        raw_path=path,
        processed_path=path,
        rows_downloaded=0,
        rows_updated=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
        metadata={"status": "skipped", "reason": reason},
    )
