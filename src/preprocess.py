import argparse
import json
import sqlite3
from pathlib import Path

from config import DEFAULT_DB_PATH, OUTPUTS_DIR, ensure_project_dirs


def execute_script(conn: sqlite3.Connection, script: str) -> None:
    conn.executescript(script)
    conn.commit()


def create_train_test_split(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        """
        DROP TABLE IF EXISTS test_events;
        CREATE TABLE test_events AS
        WITH ranked_transactions AS (
            SELECT
                event_id,
                visitorid,
                ROW_NUMBER() OVER (
                    PARTITION BY visitorid
                    ORDER BY timestamp DESC, event_id DESC
                ) AS row_number
            FROM events
            WHERE event = 'transaction'
        )
        SELECT e.*
        FROM events e
        INNER JOIN ranked_transactions r
            ON e.event_id = r.event_id
        WHERE r.row_number = 1;

        DROP TABLE IF EXISTS train_events;
        CREATE TABLE train_events AS
        SELECT e.*
        FROM events e
        LEFT JOIN test_events t
            ON e.visitorid = t.visitorid
        WHERE t.visitorid IS NULL
           OR e.timestamp < t.timestamp
           OR (e.timestamp = t.timestamp AND e.event_id < t.event_id);
        """,
    )


def create_feature_tables(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        """
        DROP TABLE IF EXISTS product_popularity;
        CREATE TABLE product_popularity AS
        SELECT
            itemid,
            COUNT(*) AS interaction_count,
            SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) AS view_count,
            SUM(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) AS addtocart_count,
            SUM(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) AS transaction_count,
            SUM(
                CASE
                    WHEN event = 'view' THEN 1.0
                    WHEN event = 'addtocart' THEN 3.0
                    WHEN event = 'transaction' THEN 5.0
                    ELSE 0.0
                END
            ) AS weighted_score,
            MAX(timestamp) AS last_timestamp
        FROM train_events
        GROUP BY itemid;

        DROP TABLE IF EXISTS user_product_scores;
        CREATE TABLE user_product_scores AS
        SELECT
            visitorid,
            itemid,
            COUNT(*) AS interaction_count,
            SUM(
                CASE
                    WHEN event = 'view' THEN 1.0
                    WHEN event = 'addtocart' THEN 3.0
                    WHEN event = 'transaction' THEN 5.0
                    ELSE 0.0
                END
            ) AS implicit_score,
            MAX(timestamp) AS last_timestamp
        FROM train_events
        GROUP BY visitorid, itemid;

        DROP TABLE IF EXISTS user_category_preferences;
        CREATE TABLE user_category_preferences AS
        SELECT
            t.visitorid,
            p.category_id AS categoryid,
            COUNT(*) AS interaction_count,
            SUM(
                CASE
                    WHEN t.event = 'view' THEN 1.0
                    WHEN t.event = 'addtocart' THEN 3.0
                    WHEN t.event = 'transaction' THEN 5.0
                    ELSE 0.0
                END
            ) AS category_score,
            MAX(t.timestamp) AS last_timestamp
        FROM train_events t
        INNER JOIN products p
            ON t.itemid = p.product_id
        WHERE p.category_id IS NOT NULL
        GROUP BY t.visitorid, p.category_id;

        DROP TABLE IF EXISTS category_product_popularity;
        CREATE TABLE category_product_popularity AS
        SELECT
            p.category_id,
            pp.itemid,
            pp.interaction_count,
            pp.view_count,
            pp.addtocart_count,
            pp.transaction_count,
            pp.weighted_score,
            pp.last_timestamp
        FROM product_popularity pp
        INNER JOIN products p
            ON pp.itemid = p.product_id
        WHERE p.category_id IS NOT NULL;
        """,
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_train_user_ts ON train_events(visitorid, timestamp);
        CREATE INDEX IF NOT EXISTS idx_train_item ON train_events(itemid);
        CREATE INDEX IF NOT EXISTS idx_test_user ON test_events(visitorid);
        CREATE INDEX IF NOT EXISTS idx_product_popularity_score ON product_popularity(weighted_score DESC);
        CREATE INDEX IF NOT EXISTS idx_user_product_user ON user_product_scores(visitorid);
        CREATE INDEX IF NOT EXISTS idx_user_category_user ON user_category_preferences(visitorid);
        CREATE INDEX IF NOT EXISTS idx_category_product_category ON category_product_popularity(category_id, weighted_score DESC);
        """,
    )


def write_preprocessing_summary(conn: sqlite3.Connection) -> None:
    ensure_project_dirs()
    summary = {
        "train_events": conn.execute("SELECT COUNT(*) FROM train_events").fetchone()[0],
        "test_events": conn.execute("SELECT COUNT(*) FROM test_events").fetchone()[0],
        "product_popularity_rows": conn.execute("SELECT COUNT(*) FROM product_popularity").fetchone()[0],
        "user_product_score_rows": conn.execute("SELECT COUNT(*) FROM user_product_scores").fetchone()[0],
        "user_category_preference_rows": conn.execute("SELECT COUNT(*) FROM user_category_preferences").fetchone()[0],
    }
    output_path = OUTPUTS_DIR / "preprocessing_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    print("Preprocessing summary")
    print("---------------------")
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    print(f"\nSaved summary to {output_path}")


def preprocess_database(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run build_database.py first.")

    conn = sqlite3.connect(db_path)
    try:
        print("Creating train/test split")
        create_train_test_split(conn)
        print("Creating feature tables")
        create_feature_tables(conn)
        print("Creating feature indexes")
        create_indexes(conn)
        conn.execute("ANALYZE")
        conn.commit()
        write_preprocessing_summary(conn)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model-ready feature tables.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    preprocess_database(args.database)


if __name__ == "__main__":
    main()
