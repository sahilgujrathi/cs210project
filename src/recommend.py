import argparse
from pathlib import Path

from config import DEFAULT_DB_PATH, DEFAULT_LATENT_MODEL_PATH
from recommender import RetailRocketRecommender


MODEL_LABELS = {
    "baseline": "Baseline popularity recommender",
    "hybrid": "Hybrid behavioral recommender",
    "hybrid_collaborative": "Hybrid collaborative recommender",
    "latent": "Latent-factor recommender",
}


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


def print_category_preferences(preferences: list[dict]) -> None:
    print("\nInferred category preferences")
    print("-----------------------------")
    if not preferences:
        print("No category preferences found for this visitor.")
        return
    for rank, row in enumerate(preferences, start=1):
        print(f"{rank:>2}. category={row['category_id']} score={row['score']:.2f}")


def print_recommendations(title: str, recommendations: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
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


def get_recommendations(
    recommender: RetailRocketRecommender,
    visitor_id: int,
    model: str,
    k: int,
) -> list[dict]:
    if model == "baseline":
        return recommender.recommend_popular(visitor_id, k=k)
    if model == "latent":
        return recommender.recommend_latent(visitor_id, k=k)
    return recommender.recommend_hybrid(
        visitor_id,
        k=k,
        include_collaborative=model == "hybrid_collaborative",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo recommendations for one RetailRocket visitor.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--latent-model", type=Path, default=DEFAULT_LATENT_MODEL_PATH)
    parser.add_argument("--visitor-id", type=int, default=None)
    parser.add_argument("--model", choices=["baseline", "hybrid", "latent"], default="hybrid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--preference-limit", type=int, default=5)
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Shows baseline, hybrid, hybrid collaborative, and latent-factor recommendations for the same visitor.",
    )
    parser.add_argument(
        "--include-collaborative",
        action="store_true",
        help="Uses the hybrid collaborative recommender when --model hybrid is selected.",
    )
    args = parser.parse_args()

    recommender = RetailRocketRecommender(args.database, latent_model_path=args.latent_model)
    try:
        visitor_id = args.visitor_id
        if visitor_id is None:
            visitor_id = recommender.choose_demo_visitor()
            if visitor_id is None:
                raise RuntimeError("Could not find a demo visitor. Did preprocessing complete?")

        selected_model = "hybrid_collaborative" if args.model == "hybrid" and args.include_collaborative else args.model

        print(f"Visitor ID: {visitor_id}")
        print(f"Demo mode: {'model comparison' if args.compare_models else MODEL_LABELS[selected_model]}")
        print()
        print_history(recommender.get_recent_history(visitor_id, limit=args.history_limit))
        print_category_preferences(recommender.get_category_preferences(visitor_id, limit=args.preference_limit))

        if args.compare_models:
            for model in ["baseline", "hybrid", "hybrid_collaborative", "latent"]:
                recommendations = get_recommendations(recommender, visitor_id, model, args.k)
                print_recommendations(MODEL_LABELS[model], recommendations)
        else:
            recommendations = get_recommendations(recommender, visitor_id, selected_model, args.k)
            print_recommendations(MODEL_LABELS[selected_model], recommendations)
    finally:
        recommender.close()


if __name__ == "__main__":
    main()
