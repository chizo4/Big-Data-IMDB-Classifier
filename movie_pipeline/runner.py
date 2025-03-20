'''
--------------------------------------------------------------
FILE:
    movie_pipeline/runner.py

INFO:
    Entry point for full pipeline setup for train/predict that handles
    command-line arguments before passing them to ClassifierPipeline.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import argparse
from datetime import datetime
import sys
from classifier_pipeline import ClassifierPipeline
from logger import get_logger


class Runner:
    '''
    -------------------------
    Runner - A class to execute the classifier pipeline.
    -------------------------
    '''

    # Base file name for results to be customized for use case.
    RESULT_FILE_BASE = r'{set_name}_{model_name}_{timestamp}.txt'

    def __init__(self: 'Runner') -> None:
        '''
        Initialize the Runner class.
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
        parser.add_argument(
            '--test-csv',
            type=str,
            required=True,
            help='Path to the prediction CSV file.'
        )
        # Optional args.
        parser.add_argument(
            '--model',
            type=str,
            default='gemma3:4b',
            help='Ollama LLM model name for genre predictor.'
        )
        return parser.parse_args()

    @staticmethod
    def _set_pred_file(base_path: str, set_name: str, model_name: str) -> str:
        '''
        Initialize the CSV prediction file for the classification task.
        Annotate it with the current timestamp and set name.

            Parameters:
            -------------------------
            base_path : str
                Base path to store results.
            set_name : str
                Name of the set to initialize the prediction file for.
            model_name : str
                Name of the LLM used for prediction.

            Returns:
            -------------------------
            pred_path : str
                Customized path name to the prediction file.
        '''
        curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_filename = Runner.RESULT_FILE_BASE
        pred_filename = pred_filename.replace('{set_name}', set_name)
        pred_filename = pred_filename.replace('{model_name}', model_name)
        pred_filename = pred_filename.replace('{timestamp}', curr_time)
        pred_path = f'{base_path}/{pred_filename}'
        return pred_path

    def _set_file_paths(self: 'Runner') -> None:
        '''
        Set the paths for the pipeline-related files.
        '''
        # Set path for model storage.
        self.model_path = self.args.model_path
        # Extract and assign data paths for the task.
        data_path = self.args.data_path
        # Extract set name identified from CSV name and model name from CLI args.
        set_name = self.args.test_csv.split('/')[-1].split('_')[0]
        model_name = self.args.model.replace(':', '_')
        self.train_cache_path = f'{data_path}/train_{model_name}_cache.csv'
        self.train_csv_path = f'{data_path}/train-*.csv'
        self.pred_cache_path = f'{data_path}/{set_name}_{model_name}_cache.csv'
        self.pred_path = self._set_pred_file(
            set_name=set_name,
            model_name=model_name,
            base_path=self.args.results_path
        )

    def __call__(self: 'Runner') -> None:
        '''
        Run the pipeline via ClassifierPipeline instance.
        '''
        self.logger.info('Creating PIPELINE files.')
        self._set_file_paths()
        self.logger.info('Running PIPELINE for movie classification.')
        pipeline = ClassifierPipeline(
            model_path=self.model_path,
            results_path=self.args.results_path,
            directing_json_path=self.args.directing_json,
            writing_json_path=self.args.writing_json,
            llm_cache_path_train=self.train_cache_path,
            llm_cache_path_test=self.pred_cache_path,
            llm_name=self.args.model
        )
        pipeline.run(
            train_csv_path=self.train_csv_path,
            test_csv_path=self.args.test_csv,
            output_txt_path=self.pred_path
        )
        self.logger.info('***SUCCESS: PIPELINE COMPLETE!***')

if __name__ == '__main__':
    pipe_runner = Runner()
    try:
        pipe_runner()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f'Error running pipeline: {str(e)}')
        logger.exception('Stack trace:')
        sys.exit(1)
