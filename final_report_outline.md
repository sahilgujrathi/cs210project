# Final Report Outline

## 1. Problem Definition and Relevance

This project studies product recommendation in e-commerce. Traditional recommenders often use only purchase history or overall product popularity. This project improves recommendations by also using browsing behavior, add-to-cart behavior, product categories, and time-based interaction patterns.

Course connection: the project combines data management, SQL database design, preprocessing, feature engineering, model evaluation, and data visualization.

## 2. Dataset Description

Dataset: RetailRocket recommender system dataset.

Main files:

- `events.csv`: visitor interactions, including `view`, `addtocart`, and `transaction`.
- `item_properties_part1.csv` and `item_properties_part2.csv`: product metadata over time.
- `category_tree.csv`: parent-child category structure.

Important dataset statistics:

- 2,756,101 behavior events.
- 2,664,312 views.
- 69,332 add-to-cart events.
- 22,457 transactions.
- 1,407,580 unique visitors.
- Event date range: 2015-05-02 to 2015-09-17.

## 3. Database Design

SQLite is used for reproducibility and simple setup.

Core tables:

- `events`: all user-product interactions.
- `users`: distinct visitors.
- `products`: distinct products and category metadata.
- `purchases`: transaction events.
- `category_tree`: product category hierarchy.
- `item_categories`: latest known category for each product.

Feature tables:

- `train_events`
- `test_events`
- `product_popularity`
- `user_product_scores`
- `user_category_preferences`
- `category_product_popularity`

## 4. Preprocessing and Feature Engineering

The data is cleaned and transformed by:

- Converting Unix millisecond timestamps into date, hour, and day-of-week features.
- Extracting latest product category values from item property snapshots.
- Assigning event weights:
  - view = 1
  - addtocart = 3
  - transaction = 5
- Creating user category preference scores.
- Creating product popularity scores.
- Splitting the final transaction for each purchasing visitor into the test set.

## 5. Recommendation Methods

### Baseline Model

The baseline recommends globally popular products based on weighted interaction counts.

### Hybrid Model

The hybrid model combines:

- Global product popularity.
- User category preferences.
- Product purchase history.
- Product recency.

This model is more aligned with the proposal because it uses browsing behavior, purchases, product metadata, and contextual behavior.

### Hybrid Collaborative Model

The hybrid collaborative model adds item co-visitation candidates from visitors who interacted with similar products. This directly addresses the proposal's collaborative filtering goal while keeping the implementation lightweight and reproducible.

## 6. Evaluation

The evaluation uses each visitor's most recent transaction as the held-out test item.

Two evaluation modes are reported:

- Purchase prediction mode allows products the user already viewed or added to cart. This measures whether the model can predict the next purchase.
- Discovery mode excludes products the user already interacted with. This measures whether the model can suggest new products.

Metrics:

- Precision@10
- Recall@10
- F1@10
- Hit Rate@10
- Category Hit Rate@10

The baseline, hybrid, and hybrid collaborative models are compared using the same test users.

Because the product catalog is large and product IDs are hashed, exact product prediction is difficult. Category Hit Rate@10 is included as an additional relevance metric that checks whether recommended products match the category of the held-out purchased item.

## 7. Results and Visualizations

Include charts from `outputs/figures/`:

- Event type distribution.
- Daily event trend.
- Events by hour of day.
- Top purchased products.
- Top product categories.
- User activity distribution.

Include the model metrics from `outputs/metrics.json`.

## 8. Discussion and Limitations

Limitations:

- The dataset contains hashed product properties, so product names and descriptions are not human-readable.
- There is no live A/B test, so evaluation is offline.
- Many visitors have very limited history, making personalization difficult.
- Cold-start users and products remain challenging.
- The model is intentionally lightweight for reproducibility.

## 9. Future Improvements

Possible extensions:

- Add a matrix factorization model.
- Add an RNN or GRU model for session-based recommendations.
- Use more item property fields beyond category.
- Improve cold-start recommendations.
- Build a web interface for the demo.

## 10. Individual Contributions

Document each group member's work:

- Data loading and database design.
- Preprocessing and feature engineering.
- Recommendation model implementation.
- Evaluation and visualizations.
- Final report and demo video.
