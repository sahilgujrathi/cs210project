# CS210 RetailRocket Recommendation System

This is my CS210 project. It uses the RetailRocket e-commerce dataset to make product recommendations for visitors.

The dataset has visitor actions like:

- viewing a product
- adding a product to the cart
- buying a product

The program puts the data into a SQLite database, makes feature tables, builds recommender models, checks how well the models work, and then runs a small demo that recommends products for one visitor.

## What The Project Does

The project has four recommendation methods:

- `baseline`: recommends the most popular products.
- `hybrid`: uses the visitor's past behavior and favorite categories.
- `hybrid_collaborative`: uses visitor behavior plus similar visitor patterns.
- `latent_factors`: uses a learned matrix model to find product patterns.

The product names are hidden in this dataset, so the project mostly uses product IDs and category IDs.

## Files Needed

The raw dataset files must be inside the `archive` folder:

```text
archive/events.csv
archive/category_tree.csv
archive/item_properties_part1.csv
archive/item_properties_part2.csv
```

The project will create these folders/files after it runs:

```text
database/retailrocket.db
outputs/analysis_summary.json
outputs/metrics.json
outputs/figures/
outputs/models/
```

## Main Project Files

```text
src/inspect_dataset.py       checks the raw CSV files
src/build_database.py        builds the SQLite database
src/preprocess.py            makes train/test data and feature tables
src/run_analysis.py          makes charts and summary stats
src/build_latent_model.py    builds the latent-factor model
src/evaluate.py              tests all recommender models
src/recommend.py             runs the recommendation demo
```

## Setup

Run these commands from the project folder:

```bash
cd "/Users/ghiriishsridharan/Documents/Codex/Project Proposal "
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After this, use `python` for the rest of the commands. If the terminal says `python` is not found, run `source .venv/bin/activate` again.

## How To Fully Test The Program

This is the full test. Run the commands in this order. It tests the whole program from raw data to final recommendations.

### 1. Check that the dataset is there

```bash
python src/inspect_dataset.py
```

This should print the number of rows, views, add-to-cart events, transactions, visitors, products, and the date range.

### 2. Build the database

```bash
python src/build_database.py
```

This creates:

```text
database/retailrocket.db
```

This step can take a while because the dataset is large. It also needs several GB of free space.

### 3. Preprocess the data

```bash
python src/preprocess.py
```

This creates the training data, test data, and feature tables. It also creates:

```text
outputs/preprocessing_summary.json
```

### 4. Make the analysis charts

```bash
python src/run_analysis.py
```

This creates charts in:

```text
outputs/figures/
```

It also creates:

```text
outputs/analysis_summary.json
```

### 5. Build the latent-factor model

```bash
python src/build_latent_model.py
```

This creates:

```text
outputs/models/latent_factors.npz
outputs/models/latent_factors_summary.json
```

### 6. Evaluate all models

```bash
python src/evaluate.py --max-users 1000 --k 10 --evaluation-mode both
```

This tests the recommendation models and saves the results to:

```text
outputs/metrics.json
```

The test uses two modes:

- `purchase_prediction`: checks if the model can guess the next item a visitor buys.
- `discovery`: checks if the model can recommend new items the visitor has not already seen.

To test every eligible user instead of only 1,000 users, run:

```bash
python src/evaluate.py --max-users 0 --k 10 --evaluation-mode both
```

### 7. Run the recommendation demo

```bash
python src/recommend.py --compare-models --k 3
```

This shows the same visitor's history and compares recommendations from all four models.

You can also test one model at a time:

```bash
python src/recommend.py --model baseline --k 5
python src/recommend.py --model hybrid --k 5
python src/recommend.py --model hybrid --include-collaborative --k 5
python src/recommend.py --model latent --k 5
```

## Quick Test

Use this if you only want to make sure the code works without building the full database. This uses a smaller sample.

```bash
python src/build_database.py --database database/retailrocket_sample.db --sample-events 50000 --sample-property-rows 300000
python src/preprocess.py --database database/retailrocket_sample.db
python src/run_analysis.py --database database/retailrocket_sample.db
python src/build_latent_model.py --database database/retailrocket_sample.db --output outputs/models/latent_factors_sample.npz --max-users 500 --max-items 500 --factors 16
python src/evaluate.py --database database/retailrocket_sample.db --latent-model outputs/models/latent_factors_sample.npz --max-users 100 --k 10 --evaluation-mode both
python src/recommend.py --database database/retailrocket_sample.db --latent-model outputs/models/latent_factors_sample.npz --compare-models --k 3
```

If all of those commands finish without errors, the program is working.

## What To Check After Testing

After the full test, these files should exist:

```text
database/retailrocket.db
outputs/preprocessing_summary.json
outputs/analysis_summary.json
outputs/metrics.json
outputs/models/latent_factors.npz
outputs/models/latent_factors_summary.json
outputs/figures/event_type_distribution.png
outputs/figures/daily_events.png
outputs/figures/events_by_hour.png
outputs/figures/top_purchased_products.png
outputs/figures/top_categories.png
outputs/figures/user_activity_distribution.png
```

The most important result file is:

```text
outputs/metrics.json
```

That file shows Precision@10, Recall@10, F1@10, Hit Rate@10, and Category Hit Rate@10 for each model.

## Current Results

From the latest 1,000-user test, the best exact next-purchase model was `latent_factors`.

```text
Purchase prediction Hit Rate@10:
baseline = 0.015
hybrid = 0.353
hybrid_collaborative = 0.322
latent_factors = 0.401
```

For category matching, the best model was `hybrid_collaborative`.

```text
Category Hit Rate@10:
baseline = 0.066
hybrid = 0.892
hybrid_collaborative = 0.902
latent_factors = 0.479
```

This means the latent-factor model was better at guessing the exact item, but the hybrid collaborative model was better at recommending products from the right category.

## Notes

- The dataset does not have real product names, prices, ratings, or search text.
- The project uses product categories because those are the clearest metadata available.
- A live A/B test is not possible because this is an offline dataset.
- The original proposal mentioned an RNN, but this version uses a lighter latent-factor model instead.
