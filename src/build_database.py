import argparse
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from config import DEFAULT_DB_PATH, RAW_DATA_DIR, ensure_project_dirs


EVENT_CHUNKSIZE = 250_000
PROPERTY_CHUNKSIZE = 500_000
MIN_FREE_BYTES_FOR_FULL_BUILD = 4 * 1024**3


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Could not find required file: {path}")


def reset_database(db_path: Path) -> sqlite3.Connection:
    ensure_project_dirs()
    if db_path.exists():
        db_path.unlink()
    return sqlite3.connect(db_path)


def check_disk_space(db_path: Path, sample_events: int | None) -> None:
    ensure_project_dirs()
    if sample_events is not None:
        return

    usage = shutil.disk_usage(db_path.parent)
    if usage.free < MIN_FREE_BYTES_FOR_FULL_BUILD:
        free_gb = usage.free / 1024**3
        needed_gb = MIN_FREE_BYTES_FOR_FULL_BUILD / 1024**3
        raise RuntimeError(
            "Not enough free disk space for the full RetailRocket database build. "
            f"Available: {free_gb:.2f} GB. Recommended minimum: {needed_gb:.1f} GB. "
            "Free disk space, then rerun this command. For a quick test, use "
            "--sample-events and --sample-property-rows."
        )


def load_events(
    conn: sqlite3.Connection,
    events_path: Path,
    sample_events: int | None,
) -> int:
    require_file(events_path)
    rows_loaded = 0
    next_event_id = 1

    for chunk in pd.read_csv(events_path, chunksize=EVENT_CHUNKSIZE):
        if sample_events is not None:
            remaining = sample_events - rows_loaded
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        if chunk.empty:
            continue

        chunk.insert(0, "event_id", range(next_event_id, next_event_id + len(chunk)))
        event_time = pd.to_datetime(chunk["timestamp"], unit="ms", utc=True)
        chunk["event_date"] = event_time.dt.strftime("%Y-%m-%d")
        chunk["event_hour"] = event_time.dt.hour.astype("int16")
        chunk["day_of_week"] = event_time.dt.dayofweek.astype("int16")
        chunk["transactionid"] = (
            chunk["transactionid"]
            .where(chunk["transactionid"].notna(), "")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

        chunk.to_sql("events", conn, if_exists="append", index=False)
        rows_loaded += len(chunk)
        next_event_id += len(chunk)
        print(f"Loaded {rows_loaded:,} event rows")

    return rows_loaded


def load_category_tree(conn: sqlite3.Connection, category_tree_path: Path) -> int:
    require_file(category_tree_path)
    categories = pd.read_csv(category_tree_path)
    categories["parentid"] = pd.to_numeric(categories["parentid"], errors="coerce")
    categories.to_sql("category_tree", conn, if_exists="replace", index=False)
    return len(categories)


def extract_item_categories(
    raw_dir: Path,
    sample_property_rows: int | None,
) -> pd.DataFrame:
    property_files = [
        raw_dir / "item_properties_part1.csv",
        raw_dir / "item_properties_part2.csv",
    ]
    category_chunks: list[pd.DataFrame] = []

    for property_path in property_files:
        require_file(property_path)
        rows_read = 0
        print(f"Scanning {property_path.name} for categoryid properties")

        for chunk in pd.read_csv(property_path, chunksize=PROPERTY_CHUNKSIZE):
            if sample_property_rows is not None:
                remaining = sample_property_rows - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)
            rows_read += len(chunk)

            categories = chunk.loc[chunk["property"].eq("categoryid"), ["timestamp", "itemid", "value"]].copy()
            if categories.empty:
                continue
            categories["categoryid"] = pd.to_numeric(categories["value"], errors="coerce")
            categories = categories.dropna(subset=["categoryid"])
            categories["categoryid"] = categories["categoryid"].astype("int64")
            category_chunks.append(categories[["itemid", "categoryid", "timestamp"]])

        print(f"  Read {rows_read:,} property rows from {property_path.name}")

    if not category_chunks:
        return pd.DataFrame(columns=["itemid", "categoryid", "category_timestamp"])

    all_categories = pd.concat(category_chunks, ignore_index=True)
    all_categories = all_categories.sort_values(["itemid", "timestamp"])
    latest_categories = all_categories.drop_duplicates("itemid", keep="last")
    latest_categories = latest_categories.rename(columns={"timestamp": "category_timestamp"})
    return latest_categories.reset_index(drop=True)


def create_indexes(conn: sqlite3.Connection) -> None:
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(visitorid, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_events_item ON events(itemid)",
        "CREATE INDEX IF NOT EXISTS idx_events_event ON events(event)",
        "CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)",
        "CREATE INDEX IF NOT EXISTS idx_item_categories_item ON item_categories(itemid)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id)",
        "CREATE INDEX IF NOT EXISTS idx_purchases_user ON purchases(user_id)",
    ]
    for statement in index_statements:
        conn.execute(statement)
    conn.commit()


def create_derived_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS users;
        CREATE TABLE users AS
        SELECT DISTINCT visitorid AS user_id
        FROM events;

        DROP TABLE IF EXISTS products;
        CREATE TABLE products AS
        SELECT DISTINCT
            e.itemid AS product_id,
            ic.categoryid AS category_id
        FROM events e
        LEFT JOIN item_categories ic
            ON e.itemid = ic.itemid;

        DROP TABLE IF EXISTS purchases;
        CREATE TABLE purchases AS
        SELECT
            event_id,
            visitorid AS user_id,
            itemid AS product_id,
            timestamp,
            event_date,
            event_hour,
            transactionid
        FROM events
        WHERE event = 'transaction';
        """
    )
    conn.commit()


def print_database_summary(conn: sqlite3.Connection) -> None:
    queries = {
        "events": "SELECT COUNT(*) FROM events",
        "users": "SELECT COUNT(*) FROM users",
        "products": "SELECT COUNT(*) FROM products",
        "purchases": "SELECT COUNT(*) FROM purchases",
        "item categories": "SELECT COUNT(*) FROM item_categories",
    }
    print("\nDatabase summary")
    print("----------------")
    for label, query in queries.items():
        count = conn.execute(query).fetchone()[0]
        print(f"{label}: {count:,}")


def build_database(
    raw_dir: Path,
    db_path: Path,
    sample_events: int | None = None,
    sample_property_rows: int | None = None,
) -> None:
    check_disk_space(db_path, sample_events)
    conn = reset_database(db_path)
    try:
        print(f"Building SQLite database at {db_path}")
        event_rows = load_events(conn, raw_dir / "events.csv", sample_events)
        category_rows = load_category_tree(conn, raw_dir / "category_tree.csv")
        print(f"Loaded {category_rows:,} category tree rows")

        item_categories = extract_item_categories(raw_dir, sample_property_rows)
        item_categories.to_sql("item_categories", conn, if_exists="replace", index=False)
        print(f"Loaded {len(item_categories):,} latest item-category rows")

        create_derived_tables(conn)
        create_indexes(conn)
        conn.execute("ANALYZE")
        conn.commit()

        print_database_summary(conn)
        print(f"\nDone. Loaded {event_rows:,} events into {db_path}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SQLite database for the CS210 project.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--sample-events",
        type=int,
        default=None,
        help="Optional row limit for a quick smoke-test database.",
    )
    parser.add_argument(
        "--sample-property-rows",
        type=int,
        default=None,
        help="Optional per-file row limit while scanning item property files.",
    )
    args = parser.parse_args()
    build_database(args.raw_dir, args.database, args.sample_events, args.sample_property_rows)


if __name__ == "__main__":
    main()
