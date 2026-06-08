# Deployment

## Streamlit

```bash
pip install -r requirements.txt
python -m src.ingestion.init_database
streamlit run src/dashboard/app.py
```

## Scheduled Update

Use `python -m src.ingestion.daily_update` from cron, a task scheduler, or GitHub Actions.

