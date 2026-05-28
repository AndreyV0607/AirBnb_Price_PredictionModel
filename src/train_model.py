from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from evaluate_model import evaluate_models, select_final_model
from preprocessing import (
    PROJECT_DIR,
    TARGET,
    build_model_dataframe,
    load_raw_data,
    save_processed_data,
)


IMAGES_DIR = PROJECT_DIR / "images"
MODELS_DIR = PROJECT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


def get_models():
    """Return the models used in the project."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            criterion="squared_error",
            max_depth=22,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.75,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=300,
            criterion="squared_error",
            max_depth=22,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.75,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.1,
            n_estimators=100,
            subsample=1.0,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            max_features=None,
            random_state=42,
        ),
        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=42,
        ),
    }


def split_features_target(model_df, target=TARGET):
    """Create train, cross-validation, and test splits from the processed dataframe."""
    X = model_df.drop(columns=target)
    y = model_df[target]

    X_train_cv, X_test, y_train_cv, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_cv, y_train, y_cv = train_test_split(
        X_train_cv,
        y_train_cv,
        test_size=0.25,
        random_state=42,
    )

    return X_train, X_cv, X_test, y_train, y_cv, y_test


def plot_price_distribution(raw_df, output_path=IMAGES_DIR / "price_distribution.png"):
    """Save a price distribution image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.histplot(raw_df["price_total"], bins=60, kde=True, color="steelblue")
    plt.title("Price distribution")
    plt.xlabel("price_total")
    plt.ylabel("Listings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def plot_feature_importance(
    model,
    feature_names,
    X=None,
    y=None,
    output_path=IMAGES_DIR / "feature_importance.png",
):
    """Save feature importance for tree-based models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Top 20 feature importances"
    elif X is not None and y is not None:
        permutation_result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        importances = permutation_result.importances_mean
        title = "Top 20 permutation importances"
    else:
        return None

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", color="seagreen")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def train_project_model():
    """Train, evaluate, and save the best project model."""
    raw_df = load_raw_data()
    model_df, model_input_df, metadata = build_model_dataframe(raw_df)
    save_processed_data(model_df)

    X_train, X_cv, X_test, y_train, y_cv, y_test = split_features_target(model_df)
    models = get_models()
    results_df = evaluate_models(models, X_train, X_cv, X_test, y_train, y_cv, y_test)

    best_model_name, finalist_results_df = select_final_model(results_df)
    best_model = models[best_model_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": X_train.columns.tolist(),
        "target": TARGET,
        "metrics": results_df,
        "finalist_metrics": finalist_results_df,
        "selection_rule": (
            "Top 2 by CV_RMSE, final choice by Test_R2 minus relative "
            "overfitting gap."
        ),
        "split_strategy": {
            "train": 0.6,
            "cross_validation": 0.2,
            "test": 0.2,
            "random_state": 42,
        },
        "metadata": metadata,
    }
    joblib.dump(model_artifact, BEST_MODEL_PATH)

    plot_price_distribution(raw_df)
    plot_feature_importance(best_model, X_train.columns, X=X_test, y=y_test)

    return results_df, BEST_MODEL_PATH


if __name__ == "__main__":
    results, model_path = train_project_model()
    print(results)
    print(f"Saved best model to: {model_path}")
