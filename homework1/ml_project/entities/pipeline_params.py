from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplitParams
from .train_model_params import TrainingParams


@dataclass()
class PipelineParams:
    input_train_data_path: str
    input_test_data_path: str
    output_model_path: str
    output_preprocessor_path: str
    metric_path: str
    predictions_path: str
    logging_config: str
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainingParams


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(file_path: str) -> PipelineParams:
    """Read and parse YAML config file to the dataclass."""
    with open(file_path) as fio:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(fio))
