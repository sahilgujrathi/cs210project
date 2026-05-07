import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_DB_PATH, DEFAULT_LATENT_MODEL_PATH, ensure_project_dirs


def load_interactions(conn: sqlite3.Connection, max_users: int, max_items: int) -> pd.DataFrame:
    query = """
        WITH top_users AS (
            SELECT visitorid
            FROM user_product_scores
            GROUP BY visitorid
            ORDER BY SUM(implicit_score) DESC, COUNT(*) DESC
            LIMIT ?
        ),
        top_items AS (
            SELECT itemid
            FROM product_popularity
            ORDER BY weighted_score DESC
            LIMIT ?
        )
        SELECT
            ups.visitorid,
            ups.itemid,
            ups.implicit_score
        FROM user_product_scores ups
        INNER JOIN top_users tu
            ON ups.visitorid = tu.visitorid
        INNER JOIN top_items ti
            ON ups.itemid = ti.itemid
    """
    return pd.read_sql_query(query, conn, params=(max_users, max_items))


def load_item_metadata(conn: sqlite3.Connection, max_items: int) -> pd.DataFrame:
    query = """
        SELECT
            pp.itemid,
            p.category_id,
            pp.weighted_score
        FROM product_popularity pp
        LEFT JOIN products p
            ON pp.itemid = p.product_id
        ORDER BY pp.weighted_score DESC
        LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=(max_items,))


def build_interaction_matrix(interactions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = np.sort(interactions["visitorid"].unique()).astype(np.int64)
    item_ids = np.sort(interactions["itemid"].unique()).astype(np.int64)
    user_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    item_index = {int(item_id): index for index, item_id in enumerate(item_ids)}

    rows = interactions["visitorid"].map(user_index).to_numpy()
    cols = interactions["itemid"].map(item_index).to_numpy()
    values = np.log1p(interactions["implicit_score"].to_numpy(dtype=np.float32))

    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    np.add.at(matrix, (rows, cols), values)

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(row_norms, 1e-6)
    return matrix, user_ids, item_ids


def randomized_svd(
    matrix: np.ndarray,
    factors: int,
    oversample: int,
    power_iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    rank = min(factors + oversample, min(matrix.shape))
    random_probe = rng.normal(size=(matrix.shape[1], rank)).astype(np.float32)

    basis_sample = matrix @ random_probe
    for _ in range(power_iterations):
        basis_sample = matrix @ (matrix.T @ basis_sample)

    q_matrix, _ = np.linalg.qr(basis_sample, mode="reduced")
    small_matrix = q_matrix.T @ matrix
    _, singular_values, vt_matrix = np.linalg.svd(small_matrix, full_matrices=False)

    selected = min(factors, vt_matrix.shape[0])
    item_factors = vt_matrix[:selected].T * np.sqrt(singular_values[:selected])
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_factors = item_factors / np.maximum(item_norms, 1e-6)
    return item_factors.astype(np.float32), singular_values[:selected].astype(np.float32)


def aligned_metadata(item_ids: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    category_by_item = {
        int(row.itemid): -1 if pd.isna(row.category_id) else int(row.category_id)
        for row in metadata.itertuples(index=False)
    }
    popularity_by_item = {
        int(row.itemid): float(row.weighted_score)
        for row in metadata.itertuples(index=False)
    }
    category_ids = np.array([category_by_item.get(int(item_id), -1) for item_id in item_ids], dtype=np.int64)
    popularity = np.array([popularity_by_item.get(int(item_id), 0.0) for item_id in item_ids], dtype=np.float32)
    return category_ids, popularity


def build_latent_model(
    db_path: Path,
    output_path: Path,
    max_users: int,
    max_items: int,
    factors: int,
    oversample: int,
    power_iterations: int,
    random_seed: int,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run build_database.py first.")

    ensure_project_dirs()
    conn = sqlite3.connect(db_path)
    try:
        interactions = load_interactions(conn, max_users, max_items)
        if interactions.empty:
            raise RuntimeError("No training interactions found. Run preprocess.py before build_latent_model.py.")

        metadata = load_item_metadata(conn, max_items)
    finally:
        conn.close()

    matrix, user_ids, item_ids = build_interaction_matrix(interactions)
    item_factors, singular_values = randomized_svd(
        matrix,
        factors=factors,
        oversample=oversample,
        power_iterations=power_iterations,
        random_seed=random_seed,
    )
    category_ids, popularity = aligned_metadata(item_ids, metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        product_ids=item_ids,
        item_factors=item_factors,
        category_ids=category_ids,
        popularity_scores=popularity,
        singular_values=singular_values,
    )

    summary = {
        "users_in_matrix": int(len(user_ids)),
        "items_in_matrix": int(len(item_ids)),
        "interaction_rows": int(len(interactions)),
        "latent_factors": int(item_factors.shape[1]),
        "max_users": max_users,
        "max_items": max_items,
        "random_seed": random_seed,
        "output_path": str(output_path),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Latent factor model built")
    print("-------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved model to {output_path}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight latent-factor recommender model.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_LATENT_MODEL_PATH)
    parser.add_argument("--max-users", type=int, default=3000)
    parser.add_argument("--max-items", type=int, default=4000)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    build_latent_model(
        db_path=args.database,
        output_path=args.output,
        max_users=args.max_users,
        max_items=args.max_items,
        factors=args.factors,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
