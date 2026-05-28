import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


OVERFITTING_PENALTY = 0.35


def regression_metrics(y_true, y_pred, y_train=None, y_train_pred=None):
    """Calculate regression metrics for model evaluation."""
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

    if y_train is not None and y_train_pred is not None:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics["Train_RMSE"] = train_rmse
        metrics["Overfit_Gap_RMSE"] = metrics["RMSE"] - train_rmse

    return metrics


def evaluate_models(models, X_train, X_cv, X_test, y_train, y_cv, y_test):
    """Fit models on train, select on cross-validation, and report test metrics."""
    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_cv_pred = model.predict(X_cv)
        y_test_pred = model.predict(X_test)

        cv_metrics = regression_metrics(
            y_cv,
            y_cv_pred,
            y_train=y_train,
            y_train_pred=y_train_pred,
        )
        test_metrics = regression_metrics(
            y_test,
            y_test_pred,
            y_train=y_train,
            y_train_pred=y_train_pred,
        )

        results.append(
            {
                "model": model_name,
                "Train_RMSE": cv_metrics["Train_RMSE"],
                "CV_MAE": cv_metrics["MAE"],
                "CV_RMSE": cv_metrics["RMSE"],
                "CV_R2": cv_metrics["R2"],
                "CV_MAPE_%": cv_metrics["MAPE_%"],
                "Test_MAE": test_metrics["MAE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
                "Test_MAPE_%": test_metrics["MAPE_%"],
                "CV_Overfit_Gap_RMSE": cv_metrics["Overfit_Gap_RMSE"],
                "Test_Overfit_Gap_RMSE": test_metrics["Overfit_Gap_RMSE"],
                "Generalization_Score": (
                    cv_metrics["RMSE"]
                    + OVERFITTING_PENALTY * max(cv_metrics["Overfit_Gap_RMSE"], 0)
                ),
            }
        )

    result_columns = [
        "model",
        "Train_RMSE",
        "CV_MAE",
        "CV_RMSE",
        "CV_R2",
        "CV_MAPE_%",
        "Test_MAE",
        "Test_RMSE",
        "Test_R2",
        "Test_MAPE_%",
        "CV_Overfit_Gap_RMSE",
        "Test_Overfit_Gap_RMSE",
        "Generalization_Score",
    ]

    return (
        pd.DataFrame(results)
        .loc[:, result_columns]
        .sort_values("CV_RMSE")
        .reset_index(drop=True)
    )


def add_final_selection_score(results_df):
    """Score finalists by explained variance and relative overfitting gap."""
    scored_df = results_df.copy()
    overfit_ratio = (
        scored_df["Test_Overfit_Gap_RMSE"].clip(lower=0)
        / scored_df["Test_RMSE"]
    )
    scored_df["Final_Selection_Score"] = scored_df["Test_R2"] - overfit_ratio
    return scored_df


def select_cv_finalists(results_df, n_finalists=2):
    """Select the best models by cross-validation RMSE."""
    finalists_df = (
        results_df.sort_values("CV_RMSE")
        .head(n_finalists)
        .reset_index(drop=True)
    )
    return add_final_selection_score(finalists_df)


def select_final_model(results_df, n_finalists=2):
    """Pick the final model from CV finalists using test R2 and overfitting gap."""
    finalists_df = select_cv_finalists(results_df, n_finalists=n_finalists)
    final_results_df = (
        finalists_df.sort_values("Final_Selection_Score", ascending=False)
        .reset_index(drop=True)
    )
    return final_results_df.loc[0, "model"], final_results_df
