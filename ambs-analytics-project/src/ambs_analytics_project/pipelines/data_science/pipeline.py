from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_test_split_node,
    logistic_regression_node,
    results,
    decision_tree_classifier_node,
    random_forest_classifier_node,
    xgboost_classifier_node,
    neural_network_classifier_node,
    lightgbm_classifier_node,
    catboost_classifier_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_node,
                inputs=[
                    "preprocessed_log_dataset",
                    "params:train_test_split.target_col",
                    "params:train_test_split.test_size",
                    "params:train_test_split.random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            node(
                func=logistic_regression_node,
                inputs=["X_train", "y_train", "params:logistic_regression"],
                outputs="logistic_model",
                name="logistic_regression_node",
            ),
            node(
                func=decision_tree_classifier_node,
                inputs=["X_train", "y_train", "params:decision_tree_classifier"],
                outputs="decision_tree_model",
                name="decision_tree_node",
            ),
            node(
                func=random_forest_classifier_node,
                inputs=["X_train", "y_train", "params:random_forest_classifier"],
                outputs="random_forest_model",
                name="random_forest_node",
            ),
            node(
                func=xgboost_classifier_node,
                inputs=["X_train", "y_train", "params:xgboost_classifier"],
                outputs="xgboost_model",
                name="xgboost_node",
            ),
            node(
                func=neural_network_classifier_node,
                inputs=["X_train", "y_train", "params:neural_network_classifier"],
                outputs="neural_network_model",
                name="neural_network_node",
            ),
            node(
                func=lightgbm_classifier_node,
                inputs=["X_train", "y_train", "params:lightgbm_classifier"],
                outputs="lightgbm_model",
                name="lightgbm_node",
            ),
            node(
                func=catboost_classifier_node,
                inputs=["X_train", "y_train", "params:catboost_classifier"],
                outputs="catboost_model",
                name="catboost_node",
            ),
            node(
                func=results,
                inputs=["logistic_model", "X_train", "y_train", "X_test", "y_test"],
                outputs=None,
                name="results_node",
            ),
        ]
    )


