import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import RAW_DATA_DIR


def inspect_events(events_path: Path) -> None:
    counts: Counter[str] = Counter()
    users: set[str] = set()
    items: set[str] = set()
    min_ts = None
    max_ts = None

    with events_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp = int(row["timestamp"])
            counts[row["event"]] += 1
            users.add(row["visitorid"])
            items.add(row["itemid"])
            min_ts = timestamp if min_ts is None or timestamp < min_ts else min_ts
            max_ts = timestamp if max_ts is None or timestamp > max_ts else max_ts

    print("RetailRocket events summary")
    print("---------------------------")
    print(f"Rows: {sum(counts.values()):,}")
    print(f"Views: {counts['view']:,}")
    print(f"Add-to-cart events: {counts['addtocart']:,}")
    print(f"Transactions: {counts['transaction']:,}")
    print(f"Unique visitors: {len(users):,}")
    print(f"Unique event items: {len(items):,}")
    if min_ts and max_ts:
        start = datetime.fromtimestamp(min_ts / 1000).date()
        end = datetime.fromtimestamp(max_ts / 1000).date()
        print(f"Date range: {start} to {end}")


def check_required_files(raw_dir: Path) -> None:
    required = [
        "events.csv",
        "category_tree.csv",
        "item_properties_part1.csv",
        "item_properties_part2.csv",
    ]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required dataset files: {missing_text}")

    print("Required files found:")
    for name in required:
        path = raw_dir / name
        print(f"  {name}: {path.stat().st_size / (1024 * 1024):.1f} MB")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the RetailRocket CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    check_required_files(args.raw_dir)
    inspect_events(args.raw_dir / "events.csv")


if __name__ == "__main__":
    main()
