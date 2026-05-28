from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "final_data_with_changes.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_data.csv"
TARGET = "price_total"
DISTRICT_MIN_COUNT = 100

MODEL_FEATURES = [
    "room_type",
    "max_guests",
    "num_bedrooms",
    "distance_city_center",
    "distance_metro",
    "attraction_index_norm",
    "restaurant_index_norm",
    "proximity_index",
    "city",
    "district",
    "day_type",
]

EXCLUDED_FEATURE_GROUPS = {
    "not_available_before_listing": [
        "cleanliness_score",
        "guest_satisfaction_score",
    ],
    "redundant_with_room_type": [
        "is_shared_room",
        "is_private_room",
    ],
    "redundant_raw_indexes": [
        "attraction_index",
        "restaurant_index",
    ],
    "redundant_city_level_features": [
        "state",
        "country_code",
        "country_name",
        "Crime_Index",
        "Safety_Index",
        "Monthly_Average_Net_salary",
        "Meal_at_Inexpensive_Restaurant",
        "Taxi_price_per_Km",
        "Monthly_Basic_Utilities",
        "Monthly_Rent_One_Bedroom_CC",
        "Monthly_Rent_One_Bedroom_OCC",
        "Monthly_Rent_Three_Bedroom_CC",
        "Monthly_Rent_Three_Bedroom_OCC",
    ],
}


def load_raw_data(file_path=RAW_DATA_PATH):
    """Load the raw Airbnb dataset and remove null rows and unnamed columns."""
    df = pd.read_csv(file_path)
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df.dropna().reset_index(drop=True)


def get_outlier_bounds(df, target=TARGET):
    """Return IQR lower and upper bounds for the target variable."""
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def remove_price_outliers(df, target=TARGET):
    """Remove price outliers using the IQR rule."""
    lower_bound, upper_bound = get_outlier_bounds(df, target)
    return df[df[target].between(lower_bound, upper_bound)].copy().reset_index(drop=True)


def select_model_input(df, features=MODEL_FEATURES, target=TARGET):
    """Select the target and practical pre-listing features used for modeling."""
    selected_columns = [target] + list(features)
    missing_columns = [col for col in selected_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Columns not found in dataframe: {missing_columns}")

    return df[selected_columns].copy()


def group_rare_categories(df, column, min_count=DISTRICT_MIN_COUNT, other_label="Other"):
    """Group infrequent categories to reduce sparse one-hot features."""
    grouped_df = df.copy()
    counts = grouped_df[column].value_counts()
    grouped_df[column] = grouped_df[column].where(
        grouped_df[column].map(counts) >= min_count,
        other_label,
    )
    return grouped_df


def get_categorical_columns(df, target=TARGET):
    """Return categorical columns from a modeling dataframe."""
    return (
        df.drop(columns=target)
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )


def get_categorical_levels(df, categorical_columns):
    """Capture categorical levels from training data for future predictions."""
    return {
        column: df[column].dropna().unique().tolist()
        for column in categorical_columns
    }


def one_hot_encode(df, categorical_columns=None, categorical_levels=None, target=TARGET):
    """One-hot encode categorical columns using optional fixed category levels."""
    encoded_df = df.copy()

    if categorical_columns is None:
        categorical_columns = get_categorical_columns(encoded_df, target)

    if categorical_levels is not None:
        for column, levels in categorical_levels.items():
            encoded_df[column] = pd.Categorical(encoded_df[column], categories=levels)

    encoded_df = pd.get_dummies(
        encoded_df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    return encoded_df


def build_model_dataframe(raw_df=None):
    """Create the final processed dataframe used by the modeling notebook."""
    if raw_df is None:
        raw_df = load_raw_data()

    df_wo_outliers = remove_price_outliers(raw_df)
    model_input_df = select_model_input(df_wo_outliers)
    model_input_df = group_rare_categories(model_input_df, "district")
    categorical_columns = get_categorical_columns(model_input_df)
    categorical_levels = get_categorical_levels(model_input_df, categorical_columns)
    model_df = one_hot_encode(model_input_df, categorical_columns)

    metadata = {
        "model_features": MODEL_FEATURES,
        "categorical_columns": categorical_columns,
        "categorical_levels": categorical_levels,
        "excluded_feature_groups": EXCLUDED_FEATURE_GROUPS,
        "rows_after_outlier_filter": len(df_wo_outliers),
    }

    return model_df, model_input_df, metadata


def save_processed_data(model_df, file_path=PROCESSED_DATA_PATH):
    """Save the processed modeling dataframe."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(file_path, index=False)
    return file_path
