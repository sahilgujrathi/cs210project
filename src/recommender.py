import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config import DEFAULT_DB_PATH, DEFAULT_LATENT_MODEL_PATH


class RetailRocketRecommender:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH, latent_model_path: Path = DEFAULT_LATENT_MODEL_PATH):
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.max_popularity = self._scalar("SELECT MAX(weighted_score) FROM product_popularity") or 1.0
        self.min_timestamp = self._scalar("SELECT MIN(timestamp) FROM train_events") or 0
        self.max_timestamp = self._scalar("SELECT MAX(timestamp) FROM train_events") or 1
        self._seen_items_cache: dict[int, set[int]] = {}
        self._category_preference_cache: dict[tuple[int, int], dict[int, float]] = {}
        self._category_candidate_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._popular_candidate_cache: dict[int, list[dict[str, Any]]] = {}
        self._collaborative_candidate_cache: dict[tuple[int, int, int], dict[int, float]] = {}
        self.latent_model_path = latent_model_path
        self.latent_product_ids: np.ndarray | None = None
        self.latent_item_factors: np.ndarray | None = None
        self.latent_category_ids: np.ndarray | None = None
        self.latent_popularity_scores: np.ndarray | None = None
        self.latent_product_index: dict[int, int] = {}
        if latent_model_path.exists():
            self._load_latent_model(latent_model_path)

    def close(self) -> None:
        self.conn.close()

    def _scalar(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        row = self.conn.execute(query, params).fetchone()
        return row[0] if row else None

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(query, params))

    @property
    def has_latent_model(self) -> bool:
        return self.latent_product_ids is not None and self.latent_item_factors is not None

    def _load_latent_model(self, model_path: Path) -> None:
        with np.load(model_path) as data:
            self.latent_product_ids = data["product_ids"].astype(np.int64)
            self.latent_item_factors = data["item_factors"].astype(np.float32)
            self.latent_category_ids = data["category_ids"].astype(np.int64)
            self.latent_popularity_scores = data["popularity_scores"].astype(np.float32)
        self.latent_product_index = {
            int(product_id): index
            for index, product_id in enumerate(self.latent_product_ids)
        }

    def get_seen_items(self, visitor_id: int) -> set[int]:
        if visitor_id in self._seen_items_cache:
            return self._seen_items_cache[visitor_id]

        rows = self._rows(
            """
            SELECT DISTINCT itemid
            FROM train_events
            WHERE visitorid = ?
            """,
            (visitor_id,),
        )
        seen_items = {int(row["itemid"]) for row in rows}
        self._seen_items_cache[visitor_id] = seen_items
        return seen_items

    def get_recent_history(self, visitor_id: int, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._rows(
            """
            SELECT event, itemid, event_date, event_hour
            FROM train_events
            WHERE visitorid = ?
            ORDER BY timestamp DESC, event_id DESC
            LIMIT ?
            """,
            (visitor_id, limit),
        )
        return [dict(row) for row in rows]

    def get_category_preferences(self, visitor_id: int, limit: int = 5) -> list[dict[str, Any]]:
        preferences = self._get_category_preferences(visitor_id, limit=limit)
        return [
            {"category_id": category_id, "score": score}
            for category_id, score in preferences.items()
        ]

    def choose_demo_visitor(self) -> int | None:
        row = self.conn.execute(
            """
            SELECT visitorid
            FROM train_events
            GROUP BY visitorid
            HAVING COUNT(*) >= 3
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """
        ).fetchone()
        return int(row["visitorid"]) if row else None

    def recommend_popular(self, visitor_id: int, k: int = 10, exclude_seen: bool = True) -> list[dict[str, Any]]:
        seen = self.get_seen_items(visitor_id) if exclude_seen else set()
        candidates = self._popular_candidates(limit=max(500, k * 50))

        recommendations = []
        for row in candidates.values():
            product_id = int(row["itemid"])
            if product_id in seen:
                continue
            recommendations.append(
                {
                    "product_id": product_id,
                    "category_id": row["category_id"],
                    "score": float(row["weighted_score"]),
                    "reason": "Popular across all users",
                }
            )
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_hybrid(
        self,
        visitor_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        include_collaborative: bool = False,
    ) -> list[dict[str, Any]]:
        history_items = self.get_seen_items(visitor_id)
        seen = history_items if exclude_seen else set()
        category_preferences = self._get_category_preferences(visitor_id)

        if not category_preferences:
            return self.recommend_popular(visitor_id, k, exclude_seen)

        candidates = self._category_candidates(category_preferences)
        candidates.update(self._popular_candidates(limit=500))

        if include_collaborative and history_items:
            collaborative = self._collaborative_candidates(visitor_id)
            for product_id, collab_score in collaborative.items():
                candidate = candidates.setdefault(product_id, {"collaborative_score": 0.0})
                candidate["collaborative_score"] = collab_score

        scored = []
        max_category_score = max(category_preferences.values()) or 1.0

        for product_id, candidate in candidates.items():
            if exclude_seen and product_id in seen:
                continue

            popularity_score = float(candidate.get("weighted_score") or 0.0)
            category_id = candidate.get("category_id")
            category_score = category_preferences.get(category_id, 0.0)
            transaction_count = float(candidate.get("transaction_count") or 0.0)
            collaborative_score = float(candidate.get("collaborative_score") or 0.0)
            last_timestamp = candidate.get("last_timestamp") or self.min_timestamp

            popularity_norm = math.log1p(popularity_score) / math.log1p(self.max_popularity)
            category_norm = category_score / max_category_score
            conversion_bonus = min(transaction_count / 5.0, 1.0)
            recency_norm = self._recency_norm(last_timestamp)
            collaborative_norm = min(collaborative_score / 10.0, 1.0)

            score = (
                0.35 * popularity_norm
                + 0.40 * category_norm
                + 0.10 * conversion_bonus
                + 0.10 * recency_norm
            )
            if include_collaborative:
                score += 0.05 * collaborative_norm

            reasons = []
            if category_score > 0:
                reasons.append(f"matches preferred category {category_id}")
            if popularity_score > 0:
                reasons.append("strong product popularity")
            if transaction_count > 0:
                reasons.append("has purchase history")
            if collaborative_score > 0:
                reasons.append("appears with products this visitor interacted with")

            scored.append(
                {
                    "product_id": product_id,
                    "category_id": category_id,
                    "score": score,
                    "reason": "; ".join(reasons) if reasons else "hybrid relevance score",
                }
            )

        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored[:k]

    def recommend_latent(self, visitor_id: int, k: int = 10, exclude_seen: bool = True) -> list[dict[str, Any]]:
        if not self.has_latent_model:
            raise RuntimeError(
                f"Latent factor model not found at {self.latent_model_path}. "
                "Run build_latent_model.py before evaluating latent_factors."
            )

        assert self.latent_item_factors is not None
        assert self.latent_product_ids is not None
        assert self.latent_category_ids is not None
        assert self.latent_popularity_scores is not None

        seen = self.get_seen_items(visitor_id) if exclude_seen else set()
        history_rows = self._rows(
            """
            SELECT itemid, implicit_score
            FROM user_product_scores
            WHERE visitorid = ?
            """,
            (visitor_id,),
        )

        user_vector = np.zeros(self.latent_item_factors.shape[1], dtype=np.float32)
        total_weight = 0.0
        for row in history_rows:
            product_id = int(row["itemid"])
            index = self.latent_product_index.get(product_id)
            if index is None:
                continue
            weight = math.log1p(float(row["implicit_score"]))
            user_vector += weight * self.latent_item_factors[index]
            total_weight += weight

        if total_weight <= 0:
            return self.recommend_popular(visitor_id, k=k, exclude_seen=exclude_seen)

        user_vector = user_vector / total_weight
        user_norm = np.linalg.norm(user_vector)
        if user_norm > 0:
            user_vector = user_vector / user_norm

        similarity_scores = self.latent_item_factors @ user_vector
        popularity_norm = self.latent_popularity_scores / max(float(self.latent_popularity_scores.max()), 1.0)
        scores = similarity_scores + 0.03 * popularity_norm
        ranked_indexes = np.argsort(-scores)

        recommendations = []
        for index in ranked_indexes:
            product_id = int(self.latent_product_ids[index])
            if exclude_seen and product_id in seen:
                continue
            category_id = int(self.latent_category_ids[index])
            recommendations.append(
                {
                    "product_id": product_id,
                    "category_id": None if category_id == -1 else category_id,
                    "score": float(scores[index]),
                    "reason": "latent user-product similarity from weighted interaction embeddings",
                }
            )
            if len(recommendations) == k:
                break

        return recommendations

    def _get_category_preferences(self, visitor_id: int, limit: int = 10) -> dict[int, float]:
        cache_key = (visitor_id, limit)
        if cache_key in self._category_preference_cache:
            return self._category_preference_cache[cache_key]

        rows = self._rows(
            """
            SELECT categoryid, category_score
            FROM user_category_preferences
            WHERE visitorid = ?
            ORDER BY category_score DESC
            LIMIT ?
            """,
            (visitor_id, limit),
        )
        preferences = {int(row["categoryid"]): float(row["category_score"]) for row in rows if row["categoryid"] is not None}
        self._category_preference_cache[cache_key] = preferences
        return preferences

    def _category_candidates(self, category_preferences: dict[int, float], per_category_limit: int = 150) -> dict[int, dict[str, Any]]:
        candidates: dict[int, dict[str, Any]] = {}
        for category_id in category_preferences:
            cache_key = (category_id, per_category_limit)
            if cache_key not in self._category_candidate_cache:
                rows = self._rows(
                    """
                    SELECT
                        category_id,
                        itemid,
                        weighted_score,
                        transaction_count,
                        last_timestamp
                    FROM category_product_popularity
                    WHERE category_id = ?
                    ORDER BY weighted_score DESC
                    LIMIT ?
                    """,
                    (category_id, per_category_limit),
                )
                self._category_candidate_cache[cache_key] = [dict(row) for row in rows]
            rows = self._category_candidate_cache[cache_key]
            for row in rows:
                candidates[int(row["itemid"])] = row.copy()
        return candidates

    def _popular_candidates(self, limit: int = 500) -> dict[int, dict[str, Any]]:
        if limit not in self._popular_candidate_cache:
            rows = self._rows(
                """
                SELECT
                    p.category_id,
                    pp.itemid,
                    pp.weighted_score,
                    pp.transaction_count,
                    pp.last_timestamp
                FROM product_popularity pp
                LEFT JOIN products p
                    ON pp.itemid = p.product_id
                ORDER BY pp.weighted_score DESC
                LIMIT ?
                """,
                (limit,),
            )
            self._popular_candidate_cache[limit] = [dict(row) for row in rows]
        rows = self._popular_candidate_cache[limit]
        return {int(row["itemid"]): row.copy() for row in rows}

    def _collaborative_seed_items(self, visitor_id: int, limit: int = 10) -> list[int]:
        rows = self._rows(
            """
            SELECT itemid
            FROM user_product_scores
            WHERE visitorid = ?
            ORDER BY implicit_score DESC, last_timestamp DESC
            LIMIT ?
            """,
            (visitor_id, limit),
        )
        return [int(row["itemid"]) for row in rows]

    def _collaborative_candidates(
        self,
        visitor_id: int,
        limit: int = 200,
        seed_limit: int = 10,
        max_similar_visitors: int = 500,
    ) -> dict[int, float]:
        cache_key = (visitor_id, limit, seed_limit)
        if cache_key in self._collaborative_candidate_cache:
            return self._collaborative_candidate_cache[cache_key]

        seed_items = self._collaborative_seed_items(visitor_id, seed_limit)
        if not seed_items:
            return {}

        placeholders = ",".join("?" for _ in seed_items)
        visitor_rows = self._rows(
            f"""
            SELECT visitorid
            FROM train_events
            WHERE itemid IN ({placeholders})
              AND visitorid != ?
            LIMIT ?
            """,
            tuple(seed_items + [visitor_id, max_similar_visitors * 4]),
        )
        similar_visitors = []
        seen_visitors = set()
        for row in visitor_rows:
            similar_visitor = int(row["visitorid"])
            if similar_visitor in seen_visitors:
                continue
            seen_visitors.add(similar_visitor)
            similar_visitors.append(similar_visitor)
            if len(similar_visitors) == max_similar_visitors:
                break

        if not similar_visitors:
            return {}

        collaborative_scores: Counter[int] = Counter()
        event_weights = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
        seed_set = set(seed_items)
        chunk_size = 200
        for start in range(0, len(similar_visitors), chunk_size):
            visitor_chunk = similar_visitors[start : start + chunk_size]
            visitor_placeholders = ",".join("?" for _ in visitor_chunk)
            interaction_rows = self._rows(
                f"""
                SELECT itemid, event
                FROM train_events
                WHERE visitorid IN ({visitor_placeholders})
                """,
                tuple(visitor_chunk),
            )
            for row in interaction_rows:
                product_id = int(row["itemid"])
                if product_id in seed_set:
                    continue
                collaborative_scores[product_id] += event_weights.get(row["event"], 0.0)

        candidates = {
            product_id: float(score)
            for product_id, score in collaborative_scores.most_common(limit)
        }
        self._collaborative_candidate_cache[cache_key] = candidates
        return candidates

    def _recency_norm(self, timestamp: int | float) -> float:
        denominator = max(self.max_timestamp - self.min_timestamp, 1)
        return max(0.0, min(1.0, (float(timestamp) - self.min_timestamp) / denominator))
