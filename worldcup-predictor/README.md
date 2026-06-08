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

The daily update now performs a no-key public Elo update:

- Downloads current international-team Elo rankings from `international-football.net`, which labels the source as `eloratings.net`.
- Saves raw HTML to `data/raw`.
- Writes cleaned rankings to `data/processed/elo_ratings_latest.csv`.
- Updates matching teams in SQLite.
- The dashboard and predictor read SQLite by default when the database exists, so updated Elo values flow into predictions.
- Keeps existing player and odds data when real providers are not configured.

Future data-source adapters are reserved for:
- World Football Elo Ratings
- FBref
- Transfermarkt
- StatsBomb
- Odds API

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
