# Airbnb Price Prediction In European Cities

This project analyzes Airbnb listing prices across European cities and builds a regression model to estimate `price_total` from practical pre-listing variables.

The goal is not only to get the lowest possible error, but to build a model that is realistic, interpretable, and defensible for a data science portfolio. For that reason, the final modeling pipeline avoids variables that would not be known before publishing a listing, such as review scores.

## Project Structure

```text
AirBnB_PricePerCity/
├── data/
│   ├── raw/
│   │   └── final_data_with_changes.csv
│   └── processed/
│       └── model_data.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── images/
├── models/
│   └── best_model.pkl
└── README.md
```

## Dataset Context

The dataset contains Airbnb listings from multiple European cities. The target variable is:

```text
price_total
```

The project explores how price changes depending on location, room type, distance to city center, distance to metro, tourism-related indexes, and other listing characteristics.

## Exploratory Data Analysis

The EDA notebook is organized into separate visual sections:

- Price distribution.
- Average price by neighborhood.
- Relationship between review-related scores and price.
- Price comparison by room type.
- Listings map using latitude and longitude.
- Correlations between numeric variables and `price_total`.

Important findings:

- Prices are right-skewed: most listings are relatively affordable, but expensive listings pull the average upward.
- Location matters strongly. City, district, and proximity/tourism variables are important signals.
- Entire homes/apartments tend to have higher prices than private or shared rooms.
- Review-related variables such as `guest_satisfaction_score` and `cleanliness_score` have weak direct correlation with price.
- Review scores were not used in the final model because a new host would not know them before publishing the listing.

## Feature Selection

The model uses practical variables that could reasonably be known before listing an Airbnb:

```text
room_type
max_guests
num_bedrooms
distance_city_center
distance_metro
attraction_index_norm
restaurant_index_norm
proximity_index
city
district
day_type
```

Some variables were excluded intentionally:

- `guest_satisfaction_score` and `cleanliness_score`: not available before the listing receives guests.
- `is_shared_room` and `is_private_room`: redundant with `room_type`.
- Raw index columns such as `attraction_index` and `restaurant_index`: redundant with normalized versions.
- Country, state, rent, salary, crime, and safety columns: mostly repeated city-level information and can introduce redundant signals.

Rare districts are grouped as `Other` to reduce sparse one-hot encoded columns and lower overfitting risk.

## Modeling Approach

The dataset is split into three parts:

```text
Train: 60%
Cross-validation: 20%
Test: 20%
```

The workflow is:

1. Train all models on the train set.
2. Compare all models using cross-validation metrics.
3. Select the best two models by `CV_RMSE`.
4. Evaluate only those two finalists on the test set.
5. Select the final model using a balance between explained variance and overfitting.

Models tested:

- Linear Regression.
- Random Forest Regressor.
- Extra Trees Regressor.
- Gradient Boosting Regressor.
- Hist Gradient Boosting Regressor.

## Metrics

The project uses the following regression metrics:

| Metric | Meaning |
|---|---|
| MAE | Average absolute error in `price_total` units. |
| RMSE | Error metric that penalizes large mistakes more heavily. |
| R2 | Percentage of price variation explained by the model. |
| MAPE | Average percentage error. |
| Overfitting Gap | Difference between validation/test RMSE and train RMSE. |

The overfitting gap is calculated as:

```text
Overfitting Gap = RMSE_validation_or_test - RMSE_train
```

For final model selection, the project uses:

```text
Final Selection Score = Test_R2 - max(Test_Overfit_Gap_RMSE, 0) / Test_RMSE
```

This rewards models that explain price well and penalizes models with a high overfitting gap.

## Final Model Selection

The two finalists selected by cross-validation were:

| Model | CV RMSE | Test RMSE | Test R2 | Test MAPE | Test Overfit Gap | Final Score |
|---|---:|---:|---:|---:|---:|---:|
| Hist Gradient Boosting Regressor | 87.09 | 85.31 | 0.772 | 23.43% | 6.17 | 0.700 |
| Random Forest Regressor | 84.95 | 82.78 | 0.786 | 22.19% | 18.49 | 0.562 |

Random Forest achieved the lowest RMSE and highest R2, which means it had the strongest raw predictive performance. However, its overfitting gap was much larger.

Hist Gradient Boosting had slightly higher error, but generalized better. Its train-validation/test gap was much smaller, making it the more stable and defensible final model.

Final selected model:

```text
Hist Gradient Boosting Regressor
```

## Conclusions

The final model explains a substantial portion of Airbnb price variation while avoiding unrealistic variables and reducing overfitting.

Main conclusions:

- Location-related features are among the strongest predictors of price.
- Room type is an important pricing factor.
- Review scores do not explain price strongly and are not appropriate for pre-listing prediction.
- Random Forest produced lower raw error but showed more overfitting.
- Hist Gradient Boosting was selected because it offered the best balance between predictive performance and generalization.

The RMSE is still relatively high because Airbnb pricing is influenced by factors not included in the dataset, such as amenities, photos, seasonality, local events, minimum nights, cancellation policy, and listing quality.

## How To Run

From the project root:

```bash
python src/train_model.py
```

This will:

- Load the raw data.
- Build the processed dataset.
- Train and evaluate the models.
- Save the best model to `models/best_model.pkl`.
- Save project images to `images/`.

## Main Artifacts

- Processed dataset: `data/processed/model_data.csv`
- Final model: `models/best_model.pkl`
- EDA notebook: `notebooks/01_eda.ipynb`
- Modeling notebook: `notebooks/02_modeling.ipynb`
- Visual outputs: `images/`
