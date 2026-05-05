import argparse
import json
import sqlite3
from pathlib import Path

from config import DEFAULT_DB_PATH, OUTPUTS_DIR, ensure_project_dirs
from recommender import RetailRocketRecommender


def load_test_cases(db_path: Path, max_users: int | None) -> list[tuple[int, int, int | None]]:
    conn = sqlite3.connect(db_path)
    try:
        limit_clause = "" if max_users is None else "LIMIT ?"
        params = () if max_users is None else (max_users,)
        rows = conn.execute(
            f"""
            SELECT
                t.visitorid,
                t.itemid,
                p.category_id
            FROM test_events t
            INNER JOIN (
                SELECT DISTINCT visitorid
                FROM train_events
            ) train_users
                ON t.visitorid = train_users.visitorid
            LEFT JOIN products p
                ON t.itemid = p.product_id
            ORDER BY t.visitorid
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [
            (
                int(visitor_id),
                int(item_id),
                int(category_id) if category_id is not None else None,
            )
            for visitor_id, item_id, category_id in rows
        ]
    finally:
        conn.close()


def evaluate_model(
    recommender: RetailRocketRecommender,
    test_cases: list[tuple[int, int, int | None]],
    model_name: str,
    k: int,
    exclude_seen: bool,
) -> dict[str, float]:
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    hit_sum = 0.0
    category_hit_sum = 0.0
    category_cases = 0
    evaluated = 0

    for visitor_id, target_item, target_category in test_cases:
        if model_name == "baseline":
            recommendations = recommender.recommend_popular(visitor_id, k=k, exclude_seen=exclude_seen)
        else:
            recommendations = recommender.recommend_hybrid(
                visitor_id,
                k=k,
                exclude_seen=exclude_seen,
                include_collaborative=model_name == "hybrid_collaborative",
            )

        recommended_ids = [row["product_id"] for row in recommendations]
        recommended_categories = {row["category_id"] for row in recommendations if row["category_id"] is not None}
        if not recommended_ids:
            continue

        hit = 1.0 if target_item in recommended_ids else 0.0
        precision = hit / k
        recall = hit
        f1 = (2 * precision * recall / (precision + recall)) if hit else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        hit_sum += hit
        if target_category is not None:
            category_cases += 1
            category_hit_sum += 1.0 if target_category in recommended_categories else 0.0
        evaluated += 1

    if evaluated == 0:
        return {
            "users_evaluated": 0,
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"f1_at_{k}": 0.0,
            f"hit_rate_at_{k}": 0.0,
            f"category_hit_rate_at_{k}": 0.0,
        }

    return {
        "users_evaluated": evaluated,
        f"precision_at_{k}": precision_sum / evaluated,
        f"recall_at_{k}": recall_sum / evaluated,
        f"f1_at_{k}": f1_sum / evaluated,
        f"hit_rate_at_{k}": hit_sum / evaluated,
        "test_cases_with_category": category_cases,
        f"category_hit_rate_at_{k}": category_hit_sum / category_cases if category_cases else 0.0,
    }


def evaluate_mode(
    recommender: RetailRocketRecommender,
    test_cases: list[tuple[int, int, int | None]],
    k: int,
    exclude_seen: bool,
) -> dict[str, object]:
    return {
        "exclude_seen_items": exclude_seen,
        "baseline": evaluate_model(recommender, test_cases, "baseline", k, exclude_seen),
        "hybrid": evaluate_model(recommender, test_cases, "hybrid", k, exclude_seen),
        "hybrid_collaborative": evaluate_model(
            recommender,
            test_cases,
            "hybrid_collaborative",
            k,
            exclude_seen,
        ),
    }


def run_evaluation(
    db_path: Path,
    max_users: int | None,
    k: int,
    evaluation_mode: str,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run build_database.py first.")

    ensure_project_dirs()
    test_cases = load_test_cases(db_path, max_users)
    if not test_cases:
        raise RuntimeError("No test cases found. Run preprocess.py before evaluate.py.")

    mode_config = {
        "purchase_prediction": False,
        "discovery": True,
    }
    selected_modes = mode_config if evaluation_mode == "both" else {evaluation_mode: mode_config[evaluation_mode]}

    recommender = RetailRocketRecommender(db_path)
    try:
        results = {
            "k": k,
            "test_cases": len(test_cases),
            "evaluation_modes": {
                mode_name: evaluate_mode(recommender, test_cases, k, exclude_seen)
                for mode_name, exclude_seen in selected_modes.items()
            },
        }
    finally:
        recommender.close()

    metrics_path = OUTPUTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    print("Evaluation results")
    print("------------------")
    print(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and hybrid recommenders.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--max-users", type=int, default=1000, help="Number of test users to evaluate. Use 0 for all.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--evaluation-mode",
        choices=["both", "purchase_prediction", "discovery"],
        default="both",
        help=(
            "purchase_prediction allows previously seen items; discovery excludes them. "
            "Use both to report both evaluation frames."
        ),
    )
    args = parser.parse_args()
    max_users = None if args.max_users == 0 else args.max_users
    run_evaluation(args.database, max_users, args.k, args.evaluation_mode)


if __name__ == "__main__":
    main()
