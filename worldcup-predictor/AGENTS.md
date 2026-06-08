# AGENTS

This repository is a production-minded MVP for football forecasting.

## Operating Principles
- Prefer stable, maintainable implementations over experimental complexity.
- Keep data-source adapters independent from feature engineering and model code.
- Preserve cross-platform compatibility for Windows, macOS, and Linux.
- Use sample CSV data for demos, but keep external-source interfaces easy to replace.

## Common Commands
- Install: `pip install -r requirements.txt`
- Initialize database: `python -m src.ingestion.init_database`
- Run dashboard: `streamlit run src/dashboard/app.py`
- Run tests: `pytest`
- Run daily update: `python -m src.ingestion.daily_update`

## Development Notes
- The default SQLite database path is `database/worldcup_predictor.sqlite`.
- Streamlit pages should work with sample data even before external integrations are configured.
- Tournament simulation defaults are intentionally conservative so the app remains responsive.

