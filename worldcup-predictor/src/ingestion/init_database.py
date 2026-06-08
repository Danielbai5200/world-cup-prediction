from __future__ import annotations

from src.ingestion.database import initialize_database


def main() -> None:
    path = initialize_database()
    print(f"Initialized database at {path}")


if __name__ == "__main__":
    main()

