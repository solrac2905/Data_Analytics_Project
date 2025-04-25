import pandas as pd
import numpy as np
from sklearn import preprocessing

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values based on predefined assumptions.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """

    # Missing Value in REASON means "Not Provided"
    df["REASON"] = df["REASON"].fillna("Not Provided")

    # Missing Value in JOB categorized as "Not Provided"
    df["JOB"] = df["JOB"].fillna("Not Provided")

    return df

def drop_top_5_percent_missing(df: pd.DataFrame, threshold_missing: float = 0.95) -> pd.DataFrame:
    """
    Drops the top 5% of rows with the most missing values across variables.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame after removing rows with the highest 5% of missing values.
    """

    # Add a column for missing count
    df["missing_count"] = df.isnull().sum(axis=1)

    # Calculate 95th percentile threshold
    threshold = df["missing_count"].quantile(threshold_missing)

    # Filter rows below or equal to the threshold
    df = df[df["missing_count"] <= threshold].copy()

    # Drop the 'missing_count' column
    df.drop(columns = ["missing_count"], inplace = True)

    return df

def flag_variables_with_high_default_diff(df: pd.DataFrame, vars_to_check: list, threshold_diff: float = 0.10) -> pd.DataFrame:
    """
    Flags variables where the default rate difference between missing and non-missing values exceeds the threshold.

    Args:
        df: Input DataFrame.
        vars_to_check: List of variables to analyze.
        threshold_diff: Threshold for default rate difference.

    Returns:
        DataFrame with additional columns flagging rows with significant default rate differences.
    """
    strong_difference_flags = []

    for var in vars_to_check:
        missing_df = df[df[var].isnull()]
        not_missing_df = df[df[var].notnull()]

        # Calculate default rates
        missing_default = missing_df['BAD'].mean() if not missing_df.empty else None
        not_missing_default = not_missing_df['BAD'].mean() if not missing_df.empty else None

        # Skip if one of the groups is empty
        if missing_default is None or not_missing_default is None:
            continue

        # Calculate difference
        diff = abs(missing_default - not_missing_default)

        if diff > threshold_diff:
            strong_difference_flags.append(var)

    # Create Flagging Column
    for col in strong_difference_flags:
        df[f'{col}_missing'] = df[col].isnull().astype(int)

    return df

def impute_missing_data(df: pd.DataFrame, cols_to_fill_median: list, cols_to_fill_zero: list) -> pd.DataFrame:
    """
    Imputes missing data with median or zero based on column groups.

    Args:
        df: Input DataFrame.
        cols_to_fill_median: List of columns to fill with median values.
        cols_to_fill_zero: List of columns to fill with zero.

    Returns:
        DataFrame with missing data imputed.
    """

    # Impute with median for specific columns
    for col in cols_to_fill_median:
        df[col] = df[col].fillna(df[col].median())

    # Impute with zero for specific columns
    for col in cols_to_fill_zero:
        df[col] = df[col].fillna(0)

    return df

def handle_outliers(df: pd.DataFrame, upper_only: list, both_ends: list, upper_only_quantile: float = 0.995, both_ends_lower_quantile: float = 0.005, both_ends_upper_quantile: float = 0.995, clage_col: str = 'CLAGE', clage_upper_quantile: float = 0.99975) -> pd.DataFrame:
    """
    Handles outliers by capping values at specific percentiles.

    Args:
        df: Input DataFrame.
        upper_only: List of columns to cap only the upper end.
        both_ends: List of columns to cap both lower and upper ends.
        upper_only_quantile: Quantile for upper capping for `upper_only` columns.
        both_ends_lower_quantile: Lower quantile for capping for `both_ends` columns.
        both_ends_upper_quantile: Upper quantile for capping for `both_ends` columns.
        clage_col: Column for which upper limit rows are removed (default is 'CLAGE').
        clage_upper_quantile: Upper quantile for removing outliers in the `clage_col` column.

    Returns:
        DataFrame with outliers handled.
    """
    # Apply upper-only capping
    for var in upper_only:
        upper = df[var].quantile(upper_only_quantile)
        df[var] = np.where(df[var] > upper, upper, df[var])

    # Apply full winsorizing for both ends
    for var in both_ends:
        lower = df[var].quantile(both_ends_lower_quantile)
        upper = df[var].quantile(both_ends_upper_quantile)
        df[var] = np.clip(df[var], lower, upper)

    # Apply upper cap removal for specified column
    upper_limit = df[clage_col].quantile(clage_upper_quantile)
    df = df[df[clage_col] <= upper_limit]

    return df

def apply_one_hot_encoding(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to specified categorical columns.

    Args:
        df: Input DataFrame.
        categorical_columns: List of categorical columns to encode.

    Returns:
        DataFrame with one-hot encoded columns.
    """
    df = pd.get_dummies(df, columns = categorical_columns, dtype = int)
    return df

def scale_numeric_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Scales numeric features using Min-Max scaling, excluding the target variable.

    Args:
        df: Input DataFrame.
        numeric_columns: List of numeric columns to scale.
        target_column: Target column to exclude from scaling.

    Returns:
        DataFrame with scaled numeric features.
    """
    # Exclude the target column from scaling and just consider numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(target_column)
    columns_to_scale = [col for col in numeric_columns if col != target_column]

    # Apply Min-Max scaling
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(df[columns_to_scale])

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df.index)

    # Combine scaled numeric columns with the rest of the DataFrame
    non_scaled_df = df.drop(columns=columns_to_scale)
    final_df = pd.concat([non_scaled_df, scaled_df], axis=1)

    return final_df

def pre_processing_raw_data(df: pd.DataFrame, params) -> pd.DataFrame:

    df = handle_missing_values(df)
    df = drop_top_5_percent_missing(df, params.drop_top_5_percent_missing.threshold_missing)
    df = flag_variables_with_high_default_diff(df, params.flag_variables_with_high_default_diff.vars_to_check, params.flag_variables_with_high_default_diff.threshold_diff)
    df = impute_missing_data(df, params.impute_missing_data.cols_to_fill_median, params.impute_missing_data.cols_to_fill_zero)
    df = handle_outliers(df, params.handle_outliers.upper_only, params.handle_outliers.both_ends, params.handle_outliers.upper_only_quantile, params.handle_outliers.both_ends_lower_quantile, params.handle_outliers.both_ends_upper_quantile, params.handle_outliers.clage_col, params.handle_outliers.clage_upper_quantile)
    df = apply_one_hot_encoding(df, params.apply_one_hot_encoding.categorical_columns)
    df = scale_numeric_features(df, params.scale_numeric_features.target_column)

    return df


