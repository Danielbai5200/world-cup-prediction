# World Cup Predictor 2026 Development Plan

## Phase 1 - Project Foundation
- Initialize repository structure.
- Add dependency manifest, README, AGENTS, pytest config, CI workflow, and gitignore.
- Establish package layout and shared configuration.

## Phase 2 - Data And Database
- Create sample CSV datasets for teams, players, matches, odds, injuries, and Elo ratings.
- Implement SQLite schema and database initialization.
- Implement source-agnostic ingestion interfaces with CSV adapters and reserved external-source clients.

## Phase 3 - Features And Models
- Implement team, attack, defense, and form feature engineering.
- Implement Elo, Poisson, Dixon-Coles, and ensemble prediction models.
- Provide a high-level match prediction service.

## Phase 4 - Tournament Simulation
- Implement Monte Carlo simulation for a 48-team World Cup style tournament.
- Return probabilities for group qualification, round of 32, round of 16, quarterfinal, semifinal, final, and champion.

## Phase 5 - Dashboard And Automation
- Build Streamlit pages for match prediction, World Cup simulation, and data center.
- Add daily update entrypoint with cron and GitHub Actions compatibility.

## Phase 6 - Tests And Validation
- Add focused pytest coverage for data loading, database initialization, models, prediction, and simulation.
- Run test suite and basic import checks.
- Commit after each completed phase.

