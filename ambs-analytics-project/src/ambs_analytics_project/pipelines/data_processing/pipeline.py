from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pre_processing_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=pre_processing_raw_data,
                inputs=[
                    "scoring_data_raw",
                    "parameters",
                ],
                outputs="preprocessed_dataset",
                name="preprocessing_raw_node",
            )
        ]
    )
