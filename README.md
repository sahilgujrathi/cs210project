# CS210 RetailRocket Recommendation System

This project builds an e-commerce product recommendation system using the RetailRocket dataset. It loads raw user behavior data, stores it in SQLite, creates model-ready feature tables, evaluates multiple recommender models, generates analysis charts, and provides a command-line demo for individual visitors.

The project is designed to match the CS210 final project expectations: data management, preprocessing, model implementation, evaluation, visualizations, and a reproducible demo.

## Project Summary

The system uses user browsing and purchase behavior to recommend products.

It supports four recommendation methods:

- `baseline`: recommends globally popular products.
- `hybrid`: combines popularity, visitor category preferences, purchase history, and recency.
- `hybrid_collaborative`: adds item co-visitation patterns from visitors with similar histories.
- `latent_factors`: learns embedding-style product vectors with randomized SVD over weighted user-item interactions.

The strongest exact next-purchase model is the latent-factor recommender. The strongest category-level recommender is the hybrid collaborative model.

## Dataset

The project expects the RetailRocket files in:

```text
archive/events.csv
archive/category_tree.csv
archive/item_properties_part1.csv
archive/item_properties_part2.csv
```

Main data used:

- `events.csv`: views, add-to-cart events, and transactions.
- `category_tree.csv`: category parent-child structure.
- `item_properties_part1.csv` and `item_properties_part2.csv`: item metadata, especially `categoryid`.

The product IDs and metadata values are anonymized. Because of that, the project uses product categories as the main interpretable product metadata.

## Project Structure

```text
archive/                 Raw RetailRocket CSV files
database/                SQLite database files
outputs/                 Generated metrics, summaries, models, and charts
outputs/figures/         EDA visualizations
outputs/models/          Latent-factor model files
src/inspect_dataset.py   Checks raw data files and prints dataset statistics
src/build_database.py    Builds the SQLite database
src/preprocess.py        Creates train/test and feature tables
src/run_analysis.py      Creates charts and engagement funnel summary
src/build_latent_model.py Builds the latent-factor recommender model
src/evaluate.py          Evaluates all recommender models
src/recommend.py         Runs the recommendation demo
```

## Setup

Run these commands from the project folder:

```bash
cd "/Users/ghiriishsridharan/Documents/Codex/Project Proposal "
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After the virtual environment is activated, use `python` for the remaining commands.

## Full Workflow

Run the steps in this order for the full project.

### Step 1: Inspect the raw dataset

```bash
python src/inspect_dataset.py
```

This checks that all required CSV files exist and prints the dataset size.

Expected information shown:

- Total events.
- Number of views.
- Number of add-to-cart events.
- Number of transactions.
- Unique visitors.
- Unique products/items.
- Date range.

This step proves that the project is using the full RetailRocket dataset.

### Step 2: Build the SQLite database

```bash
python src/build_database.py
```

This loads the raw CSV files into SQLite and creates the main database tables.

Created file:

```text
database/retailrocket.db
```

Important tables:

```text
events
category_tree
item_categories
users
products
purchases
```

This step can take time because the item property files are large. Keep several GB of free disk space available before running it.

Only rerun this step if you need to rebuild the database from the raw CSV files.

### Step 3: Preprocess data and create features

```bash
python src/preprocess.py
```

This creates train/test data and model feature tables.

Created tables:

```text
train_events
test_events
product_popularity
user_product_scores
user_category_preferences
category_product_popularity
```

What this step does:

- Holds out each purchasing visitor's latest transaction as a test item.
- Converts views, add-to-cart events, and transactions into weighted implicit feedback.
- Builds product popularity features.
- Builds visitor-product interaction scores.
- Builds visitor category preference scores.

This step is what turns the database into model-ready data.

### Step 4: Generate analysis charts and engagement funnel

```bash
python src/run_analysis.py
```

This creates exploratory data analysis outputs for the final report.

Created charts:

```text
outputs/figures/event_type_distribution.png
outputs/figures/daily_events.png
outputs/figures/events_by_hour.png
outputs/figures/top_purchased_products.png
outputs/figures/top_categories.png
outputs/figures/user_activity_distribution.png
```

Created summary:

```text
outputs/analysis_summary.json
```

The script also prints an engagement funnel:

```text
Views: 2,664,312
Add-to-cart events: 69,332
Transactions: 22,457
View to add-to-cart rate: 2.60%
View to transaction rate: 0.84%
Add-to-cart to transaction rate: 32.39%
```

This helps address the proposal's user engagement and conversion-rate discussion using the available offline dataset.

### Step 5: Build the latent-factor model

```bash
python src/build_latent_model.py
```

This builds the embedding-style recommender model.

Created files:

```text
outputs/models/latent_factors.npz
outputs/models/latent_factors_summary.json
```

The model uses randomized truncated SVD over weighted user-item interactions. It is a lightweight alternative to the RNN idea in the proposal and still represents users/products in a learned numerical space.

### Step 6: Evaluate the recommender models

```bash
python src/evaluate.py --max-users 1000 --k 10 --evaluation-mode both
```

This evaluates all four models on the same held-out test users.

Models compared:

```text
baseline
hybrid
hybrid_collaborative
latent_factors
```

Metrics reported:

```text
Precision@10
Recall@10
F1@10
Hit Rate@10
Category Hit Rate@10
```

Evaluation modes:

- `purchase_prediction`: allows products the visitor already viewed or added to cart. This tests whether the model can predict the next purchase.
- `discovery`: excludes products the visitor already interacted with. This tests whether the model can suggest new products.

Created file:

```text
outputs/metrics.json
```

To evaluate every eligible test user, run:

```bash
python src/evaluate.py --max-users 0 --k 10 --evaluation-mode both
```

### Step 7: Run the recommendation demo

```bash
python src/recommend.py --k 5
```

This automatically selects a visitor with enough history and prints:

```text
recent visitor history
inferred category preferences
recommended products
why each product was recommended
```

Useful demo commands:

```bash
python src/recommend.py --model baseline --k 5
python src/recommend.py --model hybrid --include-collaborative --k 5
python src/recommend.py --model latent --k 5
python src/recommend.py --compare-models --k 3
```

The best demo command is:

```bash
python src/recommend.py --compare-models --k 3
```

That command shows the baseline, hybrid, hybrid collaborative, and latent-factor recommenders for the same visitor.

## Current Results

Latest 1,000-user evaluation:

```text
Purchase prediction Hit Rate@10:
baseline = 0.015
hybrid = 0.353
hybrid_collaborative = 0.322
latent_factors = 0.401
```

```text
Category Hit Rate@10:
baseline = 0.066
hybrid = 0.892
hybrid_collaborative = 0.903
latent_factors = 0.479
```

Interpretation:

- The latent-factor model is best at exact next-purchase prediction.
- The hybrid collaborative model is best at recommending products from the correct category.
- Exact product prediction is difficult because RetailRocket has a large anonymized catalog.
- Category Hit Rate@10 is useful because it measures whether recommendations are relevant to the user's product area.

## Proposal Alignment

The implementation matches the proposal in these areas:

- Browsing history is used through `view` events.
- Purchase history is used through `transaction` events.
- Product metadata is used through category IDs.
- SQL storage is implemented with SQLite.
- Collaborative filtering is represented by the hybrid collaborative model.
- Embedding techniques are represented by the latent-factor model.
- Evaluation uses precision, recall, F1, hit rate, and category hit rate.
- Baseline comparison is included through the popularity recommender.
- User engagement is summarized through the event funnel and conversion rates.

Important limitations to mention:

- The dataset does not include search queries.
- The dataset does not include readable product names, descriptions, prices, or ratings.
- The project uses product category as the main interpretable metadata field.
- A live A/B test is not possible with an offline dataset, so the project uses offline baseline-vs-model evaluation.
- The RNN from the proposal is not implemented; it is listed as future work.

## Quick Smoke Test

Use this if you want to test the pipeline quickly without processing the full dataset:

```bash
python src/build_database.py --database database/retailrocket_sample.db --sample-events 50000 --sample-property-rows 300000
python src/preprocess.py --database database/retailrocket_sample.db
python src/run_analysis.py --database database/retailrocket_sample.db
python src/build_latent_model.py --database database/retailrocket_sample.db --output outputs/models/latent_factors_sample.npz --max-users 500 --max-items 500 --factors 16
python src/evaluate.py --database database/retailrocket_sample.db --latent-model outputs/models/latent_factors_sample.npz --max-users 100 --k 10
python src/recommend.py --database database/retailrocket_sample.db --latent-model outputs/models/latent_factors_sample.npz --k 5
```

For the final submission, use the full workflow commands instead of the sample commands.

## Demo Video Checklist

Show these in order:

1. Show the raw RetailRocket files in `archive/`.
2. Run or show `python src/inspect_dataset.py`.
3. Show the SQLite database tables from the build step.
4. Run or show `python src/preprocess.py`.
5. Run or show `python src/run_analysis.py` and the engagement funnel.
6. Show a few charts from `outputs/figures/`.
7. Run or show `python src/build_latent_model.py`.
8. Run or show `python src/evaluate.py --max-users 1000 --k 10 --evaluation-mode both`.
9. Run `python src/recommend.py --compare-models --k 3`.
10. Explain that RNN is future work and that latent factors are the implemented embedding-based model.

## Suggested Final Report Structure

1. Problem Definition and Motivation
2. Dataset Description
3. Database Design
4. Data Cleaning and Feature Engineering
5. Recommendation Methods
6. Evaluation Metrics
7. Results and Visualizations
8. Discussion and Limitations
9. Future Improvements
10. Individual Contributions
11. References
