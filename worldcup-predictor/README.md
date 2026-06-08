# World Cup Predictor 2026

Professional football prediction MVP focused on the 2026 FIFA World Cup. The system predicts match outcomes, likely scorelines, and Monte Carlo tournament advancement probabilities.

## Installation

```bash
cd worldcup-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingestion.init_database
```

## Local Run

```bash
streamlit run src/dashboard/app.py
```

The dashboard includes:
- Single match prediction
- World Cup simulation
- Data center for team, player, and injury views

## Data Update

The MVP uses CSV sample data from `data/sample` and loads it into SQLite.

```bash
python -m src.ingestion.daily_update
```

The daily update now performs public Elo, Transfermarkt player/squad, and optional odds updates:

- Checks FIFA's official men's ranking page for official ranking metadata, including last and next official update dates.
- Downloads current international-team Elo rankings from `international-football.net`, which labels the source as `eloratings.net`.
- Saves raw HTML to `data/raw`.
- Writes cleaned rankings to `data/processed/elo_ratings_latest.csv`.
- Updates matching teams in SQLite.
- Reads all 48 Transfermarkt national-team URLs from `data/config/team_source_mapping.csv`.
- Scrapes squad rows into `data/processed/transfermarkt_players_latest.csv` and updates the SQLite `players` table team by team.
- If `ODDS_API_KEY` is set, downloads match winner/draw/winner odds from The Odds API and writes `data/processed/match_odds_latest.csv`.
- If `POLYMARKET_SLUG` is set, downloads winner-market probabilities from Polymarket and writes `data/processed/winner_market_odds_latest.csv`.
- The dashboard and predictor read SQLite by default when the database exists, so updated Elo values flow into predictions.
- Keeps existing odds data when real providers are not configured.

Optional environment variables:

```bash
export ODDS_API_KEY="your-the-odds-api-key"
export ODDS_API_SPORT_KEY="soccer_fifa_world_cup"
export POLYMARKET_SLUG="fifa-world-cup-2026-winner"
export TRANSFERMARKT_DELAY_SECONDS="3"
export TRANSFERMARKT_MAX_TEAMS=""  # set a number for testing, empty means all mapped teams
```

Source integration plan:

- FIFA official rankings: primary source for official ranking metadata and future ranking ingestion if FIFA exposes stable ranking rows.
- OneFootball squads: planned source for full national-team squad lists after team URL mappings are configured.
- FBref team stats: planned source for attack/defense metrics after squad URL mappings are configured and request throttling is enabled.
- Transfermarkt squads: current source for squad snapshots, market value, and injury flags.
- The Odds API: optional source for match-level odds.
- Polymarket: optional source for outright winner-market probabilities.

The 48-team World Cup list lives in `data/config/world_cup_2026_teams.csv`. Team source mappings live in `data/config/team_source_mapping.csv`. The current version includes all 48 qualified teams, with OneFootball and Transfermarkt URLs populated for all 48. FBref URLs are populated for the original 10 sample teams and are reported as coverage so the remaining teams can be filled safely without guessing. Daily updates report mapping coverage so source drift can be detected.

Future data-source adapters are reserved for:
- World Football Elo Ratings
- FBref
- StatsBomb

Adapters are isolated in `src/ingestion`, so model code does not depend on a specific provider.

## Models

- Elo model: long-term team strength and win/draw/loss baseline.
- Poisson model: score probability matrix from expected goals, supporting 0:0 to 6:6.
- Dixon-Coles model: low-score correction on top of Poisson probabilities.
- Ensemble model: blends Elo, Poisson, Dixon-Coles, and odds-implied probabilities.

## Deployment

The app can run on any Streamlit-compatible host.

```bash
pip install -r requirements.txt
python -m src.ingestion.init_database
streamlit run src/dashboard/app.py
```

For scheduled updates, use cron:

```bash
15 3 * * * cd /path/to/worldcup-predictor && python -m src.ingestion.daily_update
```

GitHub Actions CI and daily-update workflows are included in `.github/workflows`.

## Testing

```bash
pytest
```

The current test target is focused on core correctness and runnability. Coverage is configured in `pytest.ini`; broaden tests as external data integrations and calibrated models mature.

## Current Scope

This is a runnable MVP with sample data for Argentina, France, England, Spain, Germany, Brazil, Portugal, Netherlands, Japan, and United States. It is structured for extension to Champions League, Euros, Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.
