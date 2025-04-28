from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_test_split_node, logistic_regression_node, roc_auc_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_node,
                inputs=[
                    "preprocessed_dataset",
                    "params:train_test_split.target_col",
                    "params:train_test_split.test_size",
                    "params:train_test_split.random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            node(
                func=logistic_regression_node,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:linear_reg.solver",
                    "params:linear_reg.class_weight",
                    "params:linear_reg.max_iter",
                    "params:linear_reg.random_state",
                ],
                outputs="logistic_model",
                name="logistic_regression_node",
            ),
            node(
                func=roc_auc_node,
                inputs=["logistic_model", "X_train", "y_train", "X_test", "y_test"],
                outputs=None,
                name="roc_auc_node",
            ),
        ]
    )
