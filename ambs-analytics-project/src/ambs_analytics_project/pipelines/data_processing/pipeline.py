from kedro.pipeline import Pipeline, node, pipeline

from .nodes import pre_processing_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            func = pre_processing_raw_data,
            inputs = {
                "df": "scoring_data_raw",
                "params": "parameters",
            },
            outputs = "preprocessed_dataset",
            name = "preprocessing_raw_node"
        )
        ]
    )

'''
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            func = drop_top_5_percent_missing,
            inputs = "input_dataset",
            outputs = "filtered_missing_dataset",
            name = "drop_high_missing_rows"
        ),
        node(
            func = handle_missing_values,
            inputs = {
                "df": "input_dataset",
                "threshold_missing": "params:threshold_missing"
            },
            outputs= "handle_nulls_dataset",
            name = "handle_missing_values"
        ),
        node(
            func = flag_variables_with_high_default_diff,
            inputs = {
                "df": "input_dataset",
                "vars_to_check": "params:vars_to_check",
                "threshold_diff": "params:threshold_diff"
            },
            outputs = "high_default_diff_dataset",
            name = "flag_high_default_diff_node"
        ),
        node(
            func = impute_missing_data,
            inputs = {
                "df": "input_dataset",
                "cols_to_fill_median": "params:cols_to_fill_median",
                "cols_to_fill_zero": "params:cols_to_fill_zero"
            },
            outputs = "impute_missing_dataset",
            name = "impute_missing_data_node"
        ),
        node(
            func = handle_outliers,
            inputs = {
                "df": "imputed_dataset",
                "upper_only": "params:upper_only",
                "both_ends": "params:both_ends",
                "upper_only_quantile": "params:upper_only_quantile",
                "both_ends_lower_quantile": "params:both_ends_lower_quantile",
                "both_ends_upper_quantile": "params:both_ends_upper_quantile",
                "clage_col": "params:clage_col",
                "clage_upper_quantile": "params:clage_upper_quantile"
            },
            outputs = "handle_outliers_dataset",
            name="handle_outliers_node"
        ),
        node(
            func = apply_one_hot_encoding,
            inputs = {
                "df": "imputed_dataset",
                "categorical_columns": "params:categorical_columns"
            },
            outputs = "encoded_dataset",
            name = "apply_one_hot_encoding_node"
        ),
        node(
            func = scale_numeric_features,
            inputs = {
                "df": "imputed_dataset",
                "target_column": "params:target_column"
            },
            outputs = "scaled_dataset",
            name = "scale_numeric_features_node"
        )
        ]
    )
'''
