'''
--------------------------------------------------------------
FILE:
    movie_pipeline/train.py

INFO:
    Entry point for model training pipeline that handles
    command-line arguments before passing them to ClassifierPipeline.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import argparse
import sys
from classifier_pipeline import ClassifierPipeline
from logger import get_logger


class TrainRunner:
    '''
    -------------------------
    TrainRunner - A class to execute the training pipeline.
    -------------------------
    '''

    def __init__(self: 'TrainRunner') -> None:
        '''
        Initialize the TrainRunner class.
        '''
        self.logger = get_logger(__name__)
        self.args = self._parse_args()

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        '''
        Parse command line arguments.

            Returns:
            --------
            args : argparse.Namespace
                Parsed CLI arguments.
        '''
        parser = argparse.ArgumentParser(description='Data for training pipeline.')
        # Core args (i.e., required).
        parser.add_argument(
            '--data-path',
            type=str,
            required=True,
            help='Base path to access the task data.'
        )
        parser.add_argument(
            '--directing-json',
            type=str,
            required=True,
            help='Path to the directing metadata JSON file.'
        )
        parser.add_argument(
            '--writing-json',
            type=str,
            required=True,
            help='Path to the writing metadata JSON file.'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            required=True,
            help='Path where the trained model will be saved.'
        )
        parser.add_argument(
            '--results-path',
            type=str,
            required=True,
            help='Path where the results will be saved.'
        )
        # Optional args.
        parser.add_argument(
            '--model',
            type=str,
            default='gemma3:4b',
            help='Ollama LLM model name for genre predictor.'
        )
        return parser.parse_args()

    def _set_file_paths(self: 'TrainRunner') -> None:
        '''
        Set the paths for the training-related files.
        '''
        # Extract and assign data paths for the task.
        data_path = self.args.data_path
        self.cache_path = f'{data_path}/train_cache.csv'
        self.train_csv_path = f'{data_path}/train-*.csv'
        # Set path for model storage.
        self.model_path = self.args.model_path

    def __call__(self: 'TrainRunner') -> None:
        '''
        Run the training pipeline.
        '''
        self.logger.info('Creating TRAINING files.')
        self._set_file_paths()
        self.logger.info('Running TRAINING for movie classification.')
        pipeline = ClassifierPipeline(
            model_path=self.model_path,
            results_path=self.args.results_path,
            directing_json_path=self.args.directing_json,
            writing_json_path=self.args.writing_json,
            llm_cache_path=self.cache_path,
            llm_name=self.args.model,
            train=True
        )
        pipeline.run_train(train_csv_path=self.train_csv_path)
        self.logger.info('***SUCCESS: TRAINING COMPLETE!***')


if __name__ == '__main__':
    train_runner = TrainRunner()
    try:
        train_runner()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f'Error running training pipeline: {str(e)}')
        logger.exception('Stack trace:')
        sys.exit(1)
