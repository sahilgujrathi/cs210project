import argparse
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

from config import DEFAULT_DB_PATH, FIGURES_DIR, OUTPUTS_DIR, ensure_project_dirs

MATPLOTLIB_CACHE_DIR = OUTPUTS_DIR / "matplotlib_cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotation: int = 0) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_line_chart(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    chart_df = df.copy()
    chart_df[x] = pd.to_datetime(chart_df[x])

    plt.figure(figsize=(13, 5.5))
    plt.plot(chart_df[x], chart_df[y], marker="o", linewidth=1.5, markersize=3)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())

    tick_count = min(10, len(chart_df))
    tick_positions = pd.date_range(chart_df[x].min(), chart_df[x].max(), periods=tick_count)
    tick_labels = [tick.strftime("%b %d, %Y") for tick in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_analysis(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run build_database.py first.")

    ensure_project_dirs()
    conn = sqlite3.connect(db_path)
    try:
        event_counts = pd.read_sql_query(
            """
            SELECT event, COUNT(*) AS event_count
            FROM events
            GROUP BY event
            ORDER BY event_count DESC
            """,
            conn,
        )
        save_bar_chart(
            event_counts,
            "event",
            "event_count",
            "RetailRocket Event Type Distribution",
            FIGURES_DIR / "event_type_distribution.png",
        )

        daily_events = pd.read_sql_query(
            """
            SELECT event_date, COUNT(*) AS event_count
            FROM events
            GROUP BY event_date
            ORDER BY event_date
            """,
            conn,
        )
        save_line_chart(
            daily_events,
            "event_date",
            "event_count",
            "Daily User Events",
            FIGURES_DIR / "daily_events.png",
        )

        hourly_events = pd.read_sql_query(
            """
            SELECT event_hour, COUNT(*) AS event_count
            FROM events
            GROUP BY event_hour
            ORDER BY event_hour
            """,
            conn,
        )
        save_bar_chart(
            hourly_events,
            "event_hour",
            "event_count",
            "Events by Hour of Day",
            FIGURES_DIR / "events_by_hour.png",
        )

        top_purchased = pd.read_sql_query(
            """
            SELECT itemid AS product_id, COUNT(*) AS purchase_count
            FROM events
            WHERE event = 'transaction'
            GROUP BY itemid
            ORDER BY purchase_count DESC
            LIMIT 20
            """,
            conn,
        )
        save_bar_chart(
            top_purchased,
            "product_id",
            "purchase_count",
            "Top 20 Purchased Products",
            FIGURES_DIR / "top_purchased_products.png",
            rotation=70,
        )

        top_categories = pd.read_sql_query(
            """
            SELECT p.category_id, COUNT(*) AS event_count
            FROM events e
            INNER JOIN products p
                ON e.itemid = p.product_id
            WHERE p.category_id IS NOT NULL
            GROUP BY p.category_id
            ORDER BY event_count DESC
            LIMIT 20
            """,
            conn,
        )
        save_bar_chart(
            top_categories,
            "category_id",
            "event_count",
            "Top 20 Product Categories by Interaction Count",
            FIGURES_DIR / "top_categories.png",
            rotation=70,
        )

        user_activity = pd.read_sql_query(
            """
            SELECT visitorid, COUNT(*) AS interaction_count
            FROM events
            GROUP BY visitorid
            """,
            conn,
        )
        plt.figure(figsize=(9, 5))
        plt.hist(user_activity["interaction_count"], bins=50, log=True)
        plt.title("User Activity Distribution")
        plt.xlabel("Interactions per Visitor")
        plt.ylabel("Number of Visitors, Log Scale")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "user_activity_distribution.png", dpi=160)
        plt.close()

        summary = {
            "event_counts": dict(zip(event_counts["event"], event_counts["event_count"])),
            "unique_visitors": int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]),
            "unique_products": int(conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]),
            "purchase_events": int(conn.execute("SELECT COUNT(*) FROM purchases").fetchone()[0]),
            "figures": sorted(path.name for path in FIGURES_DIR.glob("*.png")),
        }
        summary_path = OUTPUTS_DIR / "analysis_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print("Analysis complete")
        print("-----------------")
        for figure in summary["figures"]:
            print(f"Saved outputs/figures/{figure}")
        print(f"Saved summary to {summary_path}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA charts for the final report.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    run_analysis(args.database)


if __name__ == "__main__":
    main()
