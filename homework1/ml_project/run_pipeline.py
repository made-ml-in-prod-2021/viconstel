import logging.config

import click
import yaml
import pandas as pd

from entities import read_pipeline_params, PipelineParams
from data import read_data, split_train_val_data
from features import PreprocessingPipeline
from models import (train_model, evaluate_model, save_model,
                     predict_model, save_metrics, load_model)


logger = logging.getLogger('homework1')


def setup_logging(file_path: str) -> None:
    """Setup logger with logging conf (YAML) file."""
    with open(file_path) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def train_pipeline(params: PipelineParams) -> None:
    """Run training model pipeline."""
    # Read and split data
    data = read_data(params.input_train_data_path)
    logger.info(f'Loaded data from file: {params.input_train_data_path}')
    train_data, val_data = split_train_val_data(data, params.split_params)
    # Train data preprocessing
    features = params.feature_params.categorical_features
    features.extend(params.feature_params.numerical_features)
    x_train = train_data[features]
    y_train = train_data[params.feature_params.target]
    data_preprocessor = PreprocessingPipeline(
        params.feature_params.categorical_features,
        params.feature_params.numerical_features
    )
    data_preprocessor.fit(x_train)
    x_train = data_preprocessor.transform(x_train)
    logger.info('Train data preprocessing done')
    # Train model
    model = train_model(x_train, y_train, params.train_params)
    logger.info('Model training finished')
    # Validate model
    x_val = val_data[features]
    y_val = val_data[params.feature_params.target]
    x_val = data_preprocessor.transform(x_val)
    predictions = predict_model(model, x_val)
    metrics = evaluate_model(predictions, y_val)
    logger.info(f'Evaluated metrics on the validation subset: {metrics}')
    # Save artifacts to files
    save_model(model, params.output_model_path)
    logger.info(f'Model saved in file: {params.output_model_path}')
    data_preprocessor.save_pipeline(params.output_preprocessor_path)
    logger.info(f'Data preprocessor saved in file: '
                f'{params.output_preprocessor_path}')
    save_metrics(metrics, params.metric_path)
    logger.info(f'Metrics saved in file: {params.metric_path}')


def validate_pipeline(params: PipelineParams) -> None:
    """Run validate model pipeline."""
    # Read data
    data = read_data(params.input_test_data_path)
    logger.info(f'Loaded data from file: {params.input_test_data_path}')
    data_preprocessor = PreprocessingPipeline(
        params.feature_params.categorical_features,
        params.feature_params.numerical_features
    )
    # Load artifacts
    data_preprocessor.load_pipeline(params.output_preprocessor_path)
    logger.info(f'Loaded data preprocessor from file: '
                f'{params.output_preprocessor_path}')
    model = load_model(params.output_model_path, params.train_params)
    logger.info(f'Loaded model from file: {params.output_model_path}')
    # Make predictions and save it to CSV file
    data = data_preprocessor.transform(data)
    predictions = predict_model(model, data)
    logger.info('Get model predictions for input data')
    predictions_df = pd.DataFrame({'predictions': predictions})
    predictions_df.to_csv(params.predictions_path, index=False)
    logger.info(f'Predictions saved in file: {params.predictions_path}')


@click.command(name="train_pipeline")
@click.argument("config_path", default=r'configs\logreg_config.yml')
@click.argument("train_val", default='train')
def train_pipeline_command(config_path: str, train_val: str) -> None:
    params = read_pipeline_params(config_path)
    setup_logging(params.logging_config)
    logger.info(f'Get config file: {config_path}')
    if train_val == "train":
        logger.info("Run app in train mode")
        train_pipeline(params)
    elif train_val == "val":
        logger.info("Run app in validation mode")
        validate_pipeline(params)
    else:
        logger.warning("Argument must be `train` or `val`")


if __name__ == "__main__":
    train_pipeline_command()
