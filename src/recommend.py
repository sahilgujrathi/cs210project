import argparse
from pathlib import Path

from config import DEFAULT_DB_PATH
from recommender import RetailRocketRecommender


def print_history(history: list[dict]) -> None:
    print("Recent visitor history")
    print("----------------------")
    if not history:
        print("No history found in train_events.")
        return
    for row in history:
        print(
            f"{row['event_date']} hour={row['event_hour']:02d} "
            f"event={row['event']:<11} product={row['itemid']}"
        )


def print_recommendations(recommendations: list[dict]) -> None:
    print("\nRecommendations")
    print("---------------")
    if not recommendations:
        print("No recommendations found.")
        return
    for rank, row in enumerate(recommendations, start=1):
        category = row["category_id"] if row["category_id"] is not None else "unknown"
        print(
            f"{rank:>2}. product={row['product_id']} "
            f"category={category} score={row['score']:.4f}"
        )
        print(f"    why: {row['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo recommendations for one RetailRocket visitor.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--visitor-id", type=int, default=None)
    parser.add_argument("--model", choices=["baseline", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--include-collaborative",
        action="store_true",
        help="Adds slower item co-visitation candidates to the hybrid recommender.",
    )
    args = parser.parse_args()

    recommender = RetailRocketRecommender(args.database)
    try:
        visitor_id = args.visitor_id
        if visitor_id is None:
            visitor_id = recommender.choose_demo_visitor()
            if visitor_id is None:
                raise RuntimeError("Could not find a demo visitor. Did preprocessing complete?")

        print(f"Visitor ID: {visitor_id}")
        print(f"Model: {args.model}")
        print()
        print_history(recommender.get_recent_history(visitor_id))

        if args.model == "baseline":
            recommendations = recommender.recommend_popular(visitor_id, k=args.k)
        else:
            recommendations = recommender.recommend_hybrid(
                visitor_id,
                k=args.k,
                include_collaborative=args.include_collaborative,
            )
        print_recommendations(recommendations)
    finally:
        recommender.close()


if __name__ == "__main__":
    main()
